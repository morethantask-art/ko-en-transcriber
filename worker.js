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
function dtypeFor(modelId, device) {
  if (device === 'wasm') return { encoder_model: 'q8', decoder_model_merged: 'q8' };
  return SMALL_GPU_DTYPE && modelId.endsWith('small')
    ? SMALL_GPU_DTYPE
    : { encoder_model: 'fp32', decoder_model_merged: 'q4' };
}
// small은 fp32 인코더면 약 590MB라 폰에서 메모리 부족으로 실패한다. 4비트(약 300MB)도 정확도는 거의 같다.
const SMALL_GPU_DTYPE = { encoder_model: 'q4', decoder_model_merged: 'q4' };

// 16비트 PCM WAV를 Float32로 읽기 (워커에는 AudioContext가 없다)
async function loadSelfTestAudio() {
  const buf = await (await fetch('selftest.wav')).arrayBuffer();
  const v = new DataView(buf);
  let p = 12;
  while (p < v.byteLength - 8) {
    const id = String.fromCharCode(v.getUint8(p), v.getUint8(p + 1), v.getUint8(p + 2), v.getUint8(p + 3));
    const size = v.getUint32(p + 4, true);
    if (id === 'data') {
      const n = size / 2, out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = v.getInt16(p + 8 + i * 2, true) / 32768;
      return out;
    }
    p += 8 + size + (size & 1);
  }
  throw new Error('selftest.wav 읽기 실패');
}

async function loadModel(modelId, device, progress_callback) {
  model = await WhisperForConditionalGeneration.from_pretrained(modelId, {
    dtype: dtypeFor(modelId, device),
    device,
    progress_callback,
  });
  const gc = model.generation_config;
  tok = {
    sot: gc.decoder_start_token_id,
    ko: gc.lang_to_id['<|ko|>'],
    en: gc.lang_to_id['<|en|>'],
    transcribe: gc.task_to_id['transcribe'],
    noTs: gc.no_timestamps_token_id,
  };
}

// 샘플 음성("Hello, this is a quick test.")을 전사해서 결과가 맞는지 확인. 워밍업도 겸한다.
async function selfTest() {
  try {
    const r = await transcribe(await loadSelfTestAudio(), 'auto');
    return { ok: r.lang === 'en' && /test/i.test(r.text), text: r.text };
  } catch (err) {
    return { ok: false, text: String(err?.message || err) };
  }
}

async function load(modelId, preferred) {
  const progress_callback = (p) => {
    if (p.status === 'progress') {
      self.postMessage({ type: 'progress', file: p.file, loaded: p.loaded, total: p.total });
    }
  };

  [tokenizer, processor] = await Promise.all([
    AutoTokenizer.from_pretrained(modelId, { progress_callback }),
    AutoProcessor.from_pretrained(modelId, { progress_callback }),
  ]);

  let device = preferred !== 'wasm' && (await hasWebGPU()) ? 'webgpu' : 'wasm';
  let test = { ok: false, text: '' };
  if (device === 'webgpu') {
    try {
      await loadModel(modelId, device, progress_callback);
      self.postMessage({ type: 'status', text: 'GPU 결과 확인 중…' });
      test = await selfTest();
    } catch (err) {
      test = { ok: false, text: String(err?.message || err) };
    }
    // 기기에 따라 GPU 결과가 깨지는 경우가 있어서, 틀리면 CPU로 바꾼다
    if (!test.ok) {
      try { await model?.dispose(); } catch {}
      device = 'wasm';
      self.postMessage({ type: 'status', text: 'GPU 결과가 이상해서 CPU 모드로 바꾸는 중…' });
    }
  }
  if (device === 'wasm') {
    await loadModel(modelId, device, progress_callback);
    test = await selfTest();
  }
  self.postMessage({ type: 'ready', device, selftest: test });
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
      load(msg.model, msg.device).catch((err) => self.postMessage({ type: 'error', message: String(err?.message || err) }))
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
