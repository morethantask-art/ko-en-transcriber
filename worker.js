// Whisper 전사 워커: 한국어/영어 자동 감지 후 전사
import {
  AutoProcessor,
  AutoTokenizer,
  WhisperForConditionalGeneration,
  LogitsProcessor,
  LogitsProcessorList,
  env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6';

env.allowLocalModels = false;

let model = null;
let processor = null;
let tokenizer = null;
let tok = null; // 특수 토큰 id 모음

// 첫 디코딩 단계에서 언어 토큰을 ko/en 중 하나로 제한하고,
// 이어서 <|transcribe|><|notimestamps|>를 강제한다.
// 그러면 인코더를 한 번만 돌리면서 언어 감지와 전사를 같이 할 수 있다.
class LanguageForcer extends LogitsProcessor {
  constructor(langIds, taskId, noTsId) {
    super();
    this.langIds = langIds;
    this.taskId = taskId;
    this.noTsId = noTsId;
  }
  _call(input_ids, logits) {
    for (let i = 0; i < input_ids.length; ++i) {
      const len = input_ids[i].length;
      const d = /** @type {Float32Array} */ (logits[i].data);
      if (len === 1) {
        const keep = this.langIds.map((id) => d[id]);
        d.fill(-Infinity);
        this.langIds.forEach((id, k) => (d[id] = keep[k]));
      } else if (len === 2) {
        d.fill(-Infinity);
        d[this.taskId] = 0;
      } else if (len === 3 && this.noTsId != null) {
        d.fill(-Infinity);
        d[this.noTsId] = 0;
      }
    }
    return logits;
  }
}

async function hasWebGPU() {
  try {
    return !!(navigator.gpu && (await navigator.gpu.requestAdapter()));
  } catch {
    return false;
  }
}

// 주의: 인코더를 fp16으로 돌리면 결과가 깨진다(테스트로 확인). 항상 fp32 사용.
const DTYPES = {
  webgpu: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
  wasm: { encoder_model: 'q8', decoder_model_merged: 'q8' },
};

async function load(modelId) {
  const progress_callback = (p) => {
    if (p.status === 'progress') {
      self.postMessage({ type: 'progress', file: p.file, loaded: p.loaded, total: p.total });
    }
  };

  [tokenizer, processor] = await Promise.all([
    AutoTokenizer.from_pretrained(modelId, { progress_callback }),
    AutoProcessor.from_pretrained(modelId, { progress_callback }),
  ]);

  let device = (await hasWebGPU()) ? 'webgpu' : 'wasm';
  try {
    model = await WhisperForConditionalGeneration.from_pretrained(modelId, { dtype: DTYPES[device], device, progress_callback });
  } catch (err) {
    if (device !== 'webgpu') throw err;
    device = 'wasm'; // GPU가 안 되면 CPU로
    model = await WhisperForConditionalGeneration.from_pretrained(modelId, { dtype: DTYPES[device], device, progress_callback });
  }

  const gc = model.generation_config;
  tok = {
    sot: gc.decoder_start_token_id,
    ko: gc.lang_to_id['<|ko|>'],
    en: gc.lang_to_id['<|en|>'],
    transcribe: gc.task_to_id['transcribe'],
    noTs: gc.no_timestamps_token_id,
  };

  // 워밍업 (WebGPU 셰이더 컴파일)
  await transcribe(new Float32Array(16000), 'auto');
  self.postMessage({ type: 'ready', device });
}

async function transcribe(audio, langMode) {
  const inputs = await processor(audio);
  const langIds = langMode === 'ko' ? [tok.ko] : langMode === 'en' ? [tok.en] : [tok.ko, tok.en];
  const logits_processor = new LogitsProcessorList();
  logits_processor.push(new LanguageForcer(langIds, tok.transcribe, tok.noTs));

  const out = await model.generate({
    ...inputs,
    decoder_input_ids: [tok.sot],
    logits_processor,
    max_new_tokens: 200,
    no_repeat_ngram_size: 0,
  });
  const seq = out.tolist()[0].map(Number);
  const lang = seq[1] === tok.ko ? 'ko' : 'en';
  const text = tokenizer.decode(seq, { skip_special_tokens: true }).trim();
  return { lang, text };
}

// 작업은 하나씩 순서대로 처리
let chain = Promise.resolve();

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'load') {
    chain = chain.then(() =>
      load(msg.model).catch((err) => self.postMessage({ type: 'error', message: String(err?.message || err) }))
    );
  } else if (msg.type === 'transcribe') {
    chain = chain.then(async () => {
      try {
        const t0 = performance.now();
        const r = await transcribe(msg.audio, msg.lang);
        self.postMessage({ type: 'result', id: msg.id, ...r, ms: performance.now() - t0 });
      } catch (err) {
        self.postMessage({ type: 'result', id: msg.id, lang: null, text: '', error: String(err?.message || err) });
      }
    });
  }
};
