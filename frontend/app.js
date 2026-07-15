import { MicVAD } from "@ricky0123/vad-web";
import Vapi from "@vapi-ai/web";

const IDLE_PROMPT_AFTER_MS = 25_000;
const IDLE_END_AFTER_PROMPT_MS = 15_000;
const VOICE_UNAVAILABLE_TEXT = "The voice assistant is temporarily unavailable. Please try again shortly.";
const ENDPOINTING_PRESETS = {
  fast: {
    label: "Fast",
    deepgramEndpointing: 180,
    startWaitSeconds: 0.15,
    onPunctuationSeconds: 0.08,
    onNoPunctuationSeconds: 0.55,
    onNumberSeconds: 0.35,
    stopVoiceSeconds: 0.16,
    stopBackoffSeconds: 0.55,
    silero: {
      positiveSpeechThreshold: 0.54,
      negativeSpeechThreshold: 0.34,
      redemptionMs: 360,
      preSpeechPadMs: 140,
      minSpeechMs: 140,
    },
  },
  balanced: {
    label: "Balanced",
    deepgramEndpointing: 300,
    startWaitSeconds: 0.25,
    onPunctuationSeconds: 0.1,
    onNoPunctuationSeconds: 0.8,
    onNumberSeconds: 0.5,
    stopVoiceSeconds: 0.2,
    stopBackoffSeconds: 0.8,
    silero: {
      positiveSpeechThreshold: 0.58,
      negativeSpeechThreshold: 0.36,
      redemptionMs: 550,
      preSpeechPadMs: 180,
      minSpeechMs: 180,
    },
  },
  safe: {
    label: "Safe",
    deepgramEndpointing: 650,
    startWaitSeconds: 0.45,
    onPunctuationSeconds: 0.2,
    onNoPunctuationSeconds: 1.1,
    onNumberSeconds: 0.75,
    stopVoiceSeconds: 0.28,
    stopBackoffSeconds: 1.05,
    silero: {
      positiveSpeechThreshold: 0.62,
      negativeSpeechThreshold: 0.32,
      redemptionMs: 850,
      preSpeechPadMs: 220,
      minSpeechMs: 240,
    },
  },
};

function endpointingPreset(mode) {
  return ENDPOINTING_PRESETS[mode] || ENDPOINTING_PRESETS.balanced;
}

const state = {
  vapi: null,
  vad: null,
  callActive: false,
  starting: false,
  assistantSpeaking: false,
  endpointingMode: "balanced",
  voiceAssistantId: "",
  callStartedAt: 0,
  callEndedAt: 0,
  lastCallSeconds: 0,
  ledgerTimer: null,
  simpleTimer: null,
  idlePromptTimer: null,
  idleEndTimer: null,
  idlePrompted: false,
  lastActivityAt: 0,
  callEndReason: "manual",
  feedbackScore: null,
  lastFeedbackPayload: null,
  trace: null,
  latestLedger: null,
  latestReplay: null,
  latestEvaluation: null,
  latestBenchmark: null,
  latestSpeechEval: null,
  latestAudioCases: null,
  latestAudioEval: null,
  latestAudioManifest: null,
  latestAudioQuality: null,
  latestAudioAcceptedSet: null,
  latestAudioErrorAnalysis: null,
  latestAudioAccentSweep: null,
  latestAudioRobustness: null,
  latestPaperReport: null,
  latestStatisticsPack: null,
  latestClaimReadiness: null,
  latestExperimentPlan: null,
  latestCaseFactory: null,
  latestDraftValidation: null,
  latestSuitePromotion: null,
  latestRagLab: null,
  latestRagDraftAnswer: "",
  ragLabManualRunning: false,
  audioRecorder: null,
  audioStream: null,
  audioChunks: [],
  audioBlob: null,
  audioRecordingStartedAt: 0,
  bargeIn: freshBargeIn(),
  turnCount: 0,
  completedTurn: 0,
  lastAssistantSpeechStartAt: 0,
  lastAssistantSpeechEndAt: 0,
  userText: "",
  answer: "",
  metrics: freshMetrics(),
};

let publicConfigLoad = null;

const els = {
  voiceCallModal: document.querySelector("#voiceCallModal"),
  simpleCallButton: document.querySelector("#simpleCallButton"),
  simpleEndButton: document.querySelector("#simpleEndButton"),
  simpleCloseButton: document.querySelector("#simpleCloseButton"),
  simpleStatusDot: document.querySelector("#simpleStatusDot"),
  simpleStatusText: document.querySelector("#simpleStatusText"),
  simpleTimer: document.querySelector("#simpleTimer"),
  simpleHint: document.querySelector("#simpleHint"),
  simpleTranscript: document.querySelector("#simpleTranscript"),
  simpleAnswer: document.querySelector("#simpleAnswer"),
  simpleSetupMessage: document.querySelector("#simpleSetupMessage"),
  feedbackPanel: document.querySelector("#feedbackPanel"),
  feedbackButtons: Array.from(document.querySelectorAll("[data-feedback-score]")),
  feedbackComment: document.querySelector("#feedbackComment"),
  feedbackSubmitButton: document.querySelector("#feedbackSubmitButton"),
  feedbackThanks: document.querySelector("#feedbackThanks"),
  agentDock: document.querySelector("#agentDock"),
  agentDockPanel: document.querySelector(".agent-dock-panel"),
  agentCloseButton: document.querySelector("#agentCloseButton"),
  openAgentButtons: Array.from(document.querySelectorAll("[data-open-agent]")),
  openLabButtons: Array.from(document.querySelectorAll("[data-open-lab]")),
  vapiPublicKey: document.querySelector("#voicePublicKey"),
  toolServerUrl: document.querySelector("#toolServerUrl"),
  voiceId: document.querySelector("#voiceId"),
  endpointingButtons: Array.from(document.querySelectorAll("[data-endpointing-mode]")),
  endpointingPresetSummary: document.querySelector("#endpointingPresetSummary"),
  callButton: document.querySelector("#callButton"),
  callButtonTop: document.querySelector("#callButtonTop"),
  callLabel: document.querySelector("#callLabel"),
  connectionDot: document.querySelector("#connectionDot"),
  connectionText: document.querySelector("#connectionText"),
  vadDot: document.querySelector("#vadDot"),
  vadText: document.querySelector("#vadText"),
  meterBar: document.querySelector("#meterBar"),
  transcriptText: document.querySelector("#transcriptText"),
  answerText: document.querySelector("#answerText"),
  activeStage: document.querySelector("#activeStage"),
  toolTitle: document.querySelector("#toolTitle"),
  toolItemLabel: document.querySelector("#toolItemLabel"),
  toolAisleLabel: document.querySelector("#toolAisleLabel"),
  toolStockLabel: document.querySelector("#toolStockLabel"),
  toolMatchLabel: document.querySelector("#toolMatchLabel"),
  toolItem: document.querySelector("#toolItem"),
  toolAisle: document.querySelector("#toolAisle"),
  toolStock: document.querySelector("#toolStock"),
  toolMatch: document.querySelector("#toolMatch"),
  catalogStrip: document.querySelector("#catalogStrip"),
  catalogProductCount: document.querySelector("#catalogProductCount"),
  catalogDepartmentCount: document.querySelector("#catalogDepartmentCount"),
  catalogLowStock: document.querySelector("#catalogLowStock"),
  catalogOutStock: document.querySelector("#catalogOutStock"),
  catalogEvidence: document.querySelector("#catalogEvidence"),
  relationRows: document.querySelector("#relationRows"),
  metricTotal: document.querySelector("#metricTotal"),
  metricVad: document.querySelector("#metricVad"),
  metricStt: document.querySelector("#metricStt"),
  metricTts: document.querySelector("#metricTts"),
  endpointingModeMetric: document.querySelector("#endpointingModeMetric"),
  bargeInCount: document.querySelector("#bargeInCount"),
  bargeStopMs: document.querySelector("#bargeStopMs"),
  bargeCaptureMs: document.querySelector("#bargeCaptureMs"),
  overlapSavedMs: document.querySelector("#overlapSavedMs"),
  bargeStatus: document.querySelector("#bargeStatus"),
  waterfallRows: document.querySelector("#waterfallRows"),
  ledgerRows: document.querySelector("#ledgerRows"),
  assumptionAudio: document.querySelector("#assumptionAudio"),
  assumptionTokens: document.querySelector("#assumptionTokens"),
  turnCount: document.querySelector("#turnCount"),
  traceStatus: document.querySelector("#traceStatus"),
  traceCount: document.querySelector("#traceCount"),
  traceEventCount: document.querySelector("#traceEventCount"),
  traceToolCount: document.querySelector("#traceToolCount"),
  traceReplayScore: document.querySelector("#traceReplayScore"),
  traceTrustScore: document.querySelector("#traceTrustScore"),
  traceCost: document.querySelector("#traceCost"),
  feedbackHappyMetric: document.querySelector("#feedbackHappyMetric"),
  traceList: document.querySelector("#traceList"),
  traceReplayRows: document.querySelector("#traceReplayRows"),
  traceTrustRows: document.querySelector("#traceTrustRows"),
  benchmarkRunButton: document.querySelector("#benchmarkRunButton"),
  benchmarkLimitInput: document.querySelector("#benchmarkLimitInput"),
  benchmarkGroupSelect: document.querySelector("#benchmarkGroupSelect"),
  benchmarkCasesMetric: document.querySelector("#benchmarkCasesMetric"),
  benchmarkPassMetric: document.querySelector("#benchmarkPassMetric"),
  benchmarkRuntimeMetric: document.querySelector("#benchmarkRuntimeMetric"),
  benchmarkVoiceMetric: document.querySelector("#benchmarkVoiceMetric"),
  benchmarkCostMetric: document.querySelector("#benchmarkCostMetric"),
  benchmarkGateMetric: document.querySelector("#benchmarkGateMetric"),
  benchmarkRunId: document.querySelector("#benchmarkRunId"),
  benchmarkArtifacts: document.querySelector("#benchmarkArtifacts"),
  benchmarkGroupRows: document.querySelector("#benchmarkGroupRows"),
  benchmarkCaseRows: document.querySelector("#benchmarkCaseRows"),
  speechRunButton: document.querySelector("#speechRunButton"),
  speechLimitInput: document.querySelector("#speechLimitInput"),
  speechGroupSelect: document.querySelector("#speechGroupSelect"),
  speechCasesMetric: document.querySelector("#speechCasesMetric"),
  speechPassMetric: document.querySelector("#speechPassMetric"),
  speechWerMetric: document.querySelector("#speechWerMetric"),
  speechEntityMetric: document.querySelector("#speechEntityMetric"),
  speechVoiceMetric: document.querySelector("#speechVoiceMetric"),
  speechCostMetric: document.querySelector("#speechCostMetric"),
  speechRunId: document.querySelector("#speechRunId"),
  speechArtifacts: document.querySelector("#speechArtifacts"),
  speechConditionRows: document.querySelector("#speechConditionRows"),
  speechCaseRows: document.querySelector("#speechCaseRows"),
  audioRunButton: document.querySelector("#audioRunButton"),
  audioCaseSelect: document.querySelector("#audioCaseSelect"),
  audioReferenceInput: document.querySelector("#audioReferenceInput"),
  audioRecordButton: document.querySelector("#audioRecordButton"),
  audioSaveButton: document.querySelector("#audioSaveButton"),
  audioManifestButton: document.querySelector("#audioManifestButton"),
  audioQualityButton: document.querySelector("#audioQualityButton"),
  audioAcceptedButton: document.querySelector("#audioAcceptedButton"),
  audioErrorButton: document.querySelector("#audioErrorButton"),
  audioAccentSweepButton: document.querySelector("#audioAccentSweepButton"),
  audioStressButton: document.querySelector("#audioStressButton"),
  audioRobustnessButton: document.querySelector("#audioRobustnessButton"),
  audioTargetInput: document.querySelector("#audioTargetInput"),
  audioSpeakerInput: document.querySelector("#audioSpeakerInput"),
  audioAccentSelect: document.querySelector("#audioAccentSelect"),
  audioNoiseSelect: document.querySelector("#audioNoiseSelect"),
  audioDeviceSelect: document.querySelector("#audioDeviceSelect"),
  audioDistanceInput: document.querySelector("#audioDistanceInput"),
  audioNotesInput: document.querySelector("#audioNotesInput"),
  audioFallbackInput: document.querySelector("#audioFallbackInput"),
  audioPreview: document.querySelector("#audioPreview"),
  audioStatus: document.querySelector("#audioStatus"),
  audioRecordingsMetric: document.querySelector("#audioRecordingsMetric"),
  audioEvaluatedMetric: document.querySelector("#audioEvaluatedMetric"),
  audioPassMetric: document.querySelector("#audioPassMetric"),
  audioWerMetric: document.querySelector("#audioWerMetric"),
  audioEntityMetric: document.querySelector("#audioEntityMetric"),
  audioDeepgramMetric: document.querySelector("#audioSpeechMetric"),
  audioMultilingualMetric: document.querySelector("#audioMultilingualMetric"),
  audioSpanishPassMetric: document.querySelector("#audioSpanishPassMetric"),
  audioSurfaceWerMetric: document.querySelector("#audioSurfaceWerMetric"),
  audioCanonicalWerMetric: document.querySelector("#audioCanonicalWerMetric"),
  audioCanonicalEntityMetric: document.querySelector("#audioCanonicalEntityMetric"),
  audioDownstreamMetric: document.querySelector("#audioDownstreamMetric"),
  audioSemanticPassMetric: document.querySelector("#audioSemanticPassMetric"),
  audioSemanticRecoveredMetric: document.querySelector("#audioSemanticRecoveredMetric"),
  audioSemanticScoreMetric: document.querySelector("#audioSemanticScoreMetric"),
  audioSemanticIntentMetric: document.querySelector("#audioSemanticIntentMetric"),
  audioSemanticSlotMetric: document.querySelector("#audioSemanticSlotMetric"),
  audioSemanticCanonicalMetric: document.querySelector("#audioSemanticCanonicalMetric"),
  audioRobustnessMetric: document.querySelector("#audioRobustnessMetric"),
  audioRegressionMetric: document.querySelector("#audioRegressionMetric"),
  audioWorstVariantMetric: document.querySelector("#audioWorstVariantMetric"),
  audioWorstWerMetric: document.querySelector("#audioWorstWerMetric"),
  audioCoverageMetric: document.querySelector("#audioCoverageMetric"),
  audioCompleteMetric: document.querySelector("#audioCompleteMetric"),
  audioSpeakersMetric: document.querySelector("#audioSpeakersMetric"),
  audioConditionsMetric: document.querySelector("#audioConditionsMetric"),
  audioMissingMetric: document.querySelector("#audioMissingMetric"),
  audioManifestMetric: document.querySelector("#audioManifestMetric"),  audioRunId: document.querySelector("#audioRunId"),
  audioUsableMetric: document.querySelector("#audioUsableMetric"),
  audioRetakeMetric: document.querySelector("#audioRetakeMetric"),
  audioUrgentMetric: document.querySelector("#audioUrgentMetric"),
  audioEmptyMetric: document.querySelector("#audioEmptyMetric"),
  audioWrongPromptMetric: document.querySelector("#audioWrongPromptMetric"),
  audioQualityScoreMetric: document.querySelector("#audioQualityScoreMetric"),
  audioAcceptedMetric: document.querySelector("#audioAcceptedMetric"),
  audioAcceptedPassMetric: document.querySelector("#audioAcceptedPassMetric"),
  audioSupersededMetric: document.querySelector("#audioSupersededMetric"),
  audioAcceptedRetakeMetric: document.querySelector("#audioAcceptedRetakeMetric"),
  audioAcceptedWerMetric: document.querySelector("#audioAcceptedWerMetric"),
  audioAcceptedLiftMetric: document.querySelector("#audioAcceptedLiftMetric"),
  audioErrorTopMetric: document.querySelector("#audioErrorTopMetric"),
  audioErrorBucketMetric: document.querySelector("#audioErrorBucketMetric"),
  audioAsrOnlyMetric: document.querySelector("#audioAsrOnlyMetric"),
  audioDownstreamFailMetric: document.querySelector("#audioDownstreamFailMetric"),
  audioLanguageMismatchMetric: document.querySelector("#audioLanguageMismatchMetric"),
  audioCoverageGapMetric: document.querySelector("#audioCoverageGapMetric"),
  audioAccentBestMetric: document.querySelector("#audioAccentBestMetric"),
  audioAccentLiftMetric: document.querySelector("#audioAccentLiftMetric"),
  audioAccentEntityMetric: document.querySelector("#audioAccentEntityMetric"),
  audioAccentCasesMetric: document.querySelector("#audioAccentCasesMetric"),
  audioArtifacts: document.querySelector("#audioArtifacts"),
  audioManifestArtifacts: document.querySelector("#audioManifestArtifacts"),
  audioQualityArtifacts: document.querySelector("#audioQualityArtifacts"),
  audioAcceptedArtifacts: document.querySelector("#audioAcceptedArtifacts"),
  audioErrorArtifacts: document.querySelector("#audioErrorArtifacts"),
  audioAccentArtifacts: document.querySelector("#audioAccentArtifacts"),
  audioRobustnessArtifacts: document.querySelector("#audioRobustnessArtifacts"),
  audioManifestRows: document.querySelector("#audioManifestRows"),  audioRecordingRows: document.querySelector("#audioRecordingRows"),
  audioCaseRows: document.querySelector("#audioCaseRows"),
  audioQualityRows: document.querySelector("#audioQualityRows"),
  audioAcceptedRows: document.querySelector("#audioAcceptedRows"),
  audioErrorRows: document.querySelector("#audioErrorRows"),
  audioAccentRows: document.querySelector("#audioAccentRows"),
  audioRobustnessRows: document.querySelector("#audioRobustnessRows"),
  reportRunButton: document.querySelector("#reportRunButton"),
  statsRunButton: document.querySelector("#statsRunButton"),
  claimsRunButton: document.querySelector("#claimsRunButton"),
  planRunButton: document.querySelector("#planRunButton"),
  caseFactoryRunButton: document.querySelector("#caseFactoryRunButton"),
  draftValidationRunButton: document.querySelector("#draftValidationRunButton"),
  promotionRunButton: document.querySelector("#promotionRunButton"),
  reportRerunInput: document.querySelector("#reportRerunInput"),
  promotionWriteInput: document.querySelector("#promotionWriteInput"),
  reportReadinessMetric: document.querySelector("#reportReadinessMetric"),
  reportCasesMetric: document.querySelector("#reportCasesMetric"),
  reportTaskPassMetric: document.querySelector("#reportTaskPassMetric"),
  reportSpeechPassMetric: document.querySelector("#reportSpeechPassMetric"),
  reportCostMetric: document.querySelector("#reportCostMetric"),
  reportVoiceMetric: document.querySelector("#reportVoiceMetric"),
  reportRunId: document.querySelector("#reportRunId"),
  reportArtifacts: document.querySelector("#reportArtifacts"),
  reportCoreRows: document.querySelector("#reportCoreRows"),
  reportCostRows: document.querySelector("#reportCostRows"),
  statsMetricsMetric: document.querySelector("#statsMetricsMetric"),
  statsCoverageMetric: document.querySelector("#statsCoverageMetric"),
  statsWidestMetric: document.querySelector("#statsWidestMetric"),
  statsCiMetric: document.querySelector("#statsCiMetric"),
  statsArtifacts: document.querySelector("#statsArtifacts"),
  statsRows: document.querySelector("#statsRows"),
  claimReadyMetric: document.querySelector("#claimReadyMetric"),
  claimDataMetric: document.querySelector("#claimDataMetric"),
  claimWorkMetric: document.querySelector("#claimWorkMetric"),
  claimActionMetric: document.querySelector("#claimActionMetric"),
  claimsArtifacts: document.querySelector("#claimsArtifacts"),
  claimsRows: document.querySelector("#claimsRows"),
  planSamplesMetric: document.querySelector("#planSamplesMetric"),
  planTextMetric: document.querySelector("#planTextMetric"),
  planAudioMetric: document.querySelector("#planAudioMetric"),
  planProviderMetric: document.querySelector("#planProviderMetric"),
  planArtifacts: document.querySelector("#planArtifacts"),
  planRows: document.querySelector("#planRows"),
  caseFactoryTotalMetric: document.querySelector("#caseFactoryTotalMetric"),
  caseFactoryBenchmarkMetric: document.querySelector("#caseFactoryBenchmarkMetric"),
  caseFactorySpeechMetric: document.querySelector("#caseFactorySpeechMetric"),
  caseFactoryAudioMetric: document.querySelector("#caseFactoryAudioMetric"),
  caseFactoryArtifacts: document.querySelector("#caseFactoryArtifacts"),
  caseFactoryRows: document.querySelector("#caseFactoryRows"),
  draftReadyMetric: document.querySelector("#draftReadyMetric"),
  draftBlockedMetric: document.querySelector("#draftBlockedMetric"),
  draftBenchmarkMetric: document.querySelector("#draftBenchmarkMetric"),
  draftSpeechMetric: document.querySelector("#draftSpeechMetric"),
  draftValidationArtifacts: document.querySelector("#draftValidationArtifacts"),
  draftValidationRows: document.querySelector("#draftValidationRows"),
  promotionModeMetric: document.querySelector("#promotionModeMetric"),
  promotionAddedMetric: document.querySelector("#promotionAddedMetric"),
  promotionSkippedMetric: document.querySelector("#promotionSkippedMetric"),
  promotionWrittenMetric: document.querySelector("#promotionWrittenMetric"),
  promotionArtifacts: document.querySelector("#promotionArtifacts"),
  promotionRows: document.querySelector("#promotionRows"),
  ragRunButton: document.querySelector("#ragRunButton"),
  ragQueryInput: document.querySelector("#ragQueryInput"),
  ragSampleRows: document.querySelector("#ragSampleRows"),
  ragIntentMetric: document.querySelector("#ragIntentMetric"),
  ragConfidenceMetric: document.querySelector("#ragConfidenceMetric"),
  ragMarginMetric: document.querySelector("#ragMarginMetric"),
  ragValidationMetric: document.querySelector("#ragValidationMetric"),
  ragLatencyMetric: document.querySelector("#ragLatencyMetric"),
  ragSourcesMetric: document.querySelector("#ragSourcesMetric"),
  ragGateStatus: document.querySelector("#ragGateStatus"),
  ragGateAction: document.querySelector("#ragGateAction"),
  ragGateScore: document.querySelector("#ragGateScore"),
  ragGateLatency: document.querySelector("#ragGateLatency"),
  ragGateRows: document.querySelector("#ragGateRows"),
  ragPlanRows: document.querySelector("#ragPlanRows"),
  ragEvidenceRows: document.querySelector("#ragEvidenceRows"),
  ragContractRows: document.querySelector("#ragContractRows"),
  ragAblationRows: document.querySelector("#ragAblationRows"),
  ragAnswerInput: document.querySelector("#ragAnswerInput"),
  ragFaithfulnessButton: document.querySelector("#ragFaithfulnessButton"),
  ragFaithfulnessVerdict: document.querySelector("#ragFaithfulnessVerdict"),
  ragFaithfulnessScore: document.querySelector("#ragFaithfulnessScore"),
  ragFaithfulnessClaims: document.querySelector("#ragFaithfulnessClaims"),
  ragFaithfulnessSignatures: document.querySelector("#ragFaithfulnessSignatures"),
  ragFaithfulnessRows: document.querySelector("#ragFaithfulnessRows"),
};

const sampleItems = ["paper towels", "ground beef", "baby wipes", "toothpaste", "frozen pizza", "dog food"];
const sampleRagQuestions = [
  "The shelf is empty but your system shows stock.",
  "Can I bring back an open electric item?",
  "My curb side order says ready where do I go?",
  "The app price is different from shelf price.",
  "Do you have a wheel chair cart near the entrance?",
  "Can I buy a fishing license here?",
];

const audioStressVariants = [
  {
    type: "store_noise",
    label: "Synthetic store noise",
    playbackRate: 1,
    speechGain: 0.88,
    noiseAmplitude: 0.035,
    beepAmplitude: 0.018,
    metadata: { noise: "synthetic_store_noise", device: "browser_mic_augmented", environment: "audio_stress_lab" },
  },
  {
    type: "low_volume",
    label: "Low volume",
    playbackRate: 1,
    speechGain: 0.34,
    metadata: { noise: "low_volume", device: "browser_mic_augmented", environment: "audio_stress_lab" },
  },
  {
    type: "fast_speech",
    label: "Fast speech",
    playbackRate: 1.18,
    speechGain: 0.95,
    metadata: { noise: "rate_shift_fast", device: "browser_mic_augmented", environment: "audio_stress_lab" },
  },
  {
    type: "slow_speech",
    label: "Slow speech",
    playbackRate: 0.84,
    speechGain: 0.95,
    metadata: { noise: "rate_shift_slow", device: "browser_mic_augmented", environment: "audio_stress_lab" },
  },
  {
    type: "clipped_audio",
    label: "Mild clipping",
    playbackRate: 1,
    speechGain: 1.65,
    clipAt: 0.72,
    metadata: { noise: "mild_clipping", device: "browser_mic_augmented", environment: "audio_stress_lab" },
  },
];


function freshMetrics() {
  return {
    turn: 0,
    speechStartedAt: 0,
    speechEndedAt: 0,
    userFinalAt: 0,
    assistantFirstAt: 0,
    voiceStartedAt: 0,
    toolStartedAt: 0,
    toolEndedAt: 0,
    assistantSpeechEndedAt: 0,
    endpointingMode: "balanced",
    endpointingTargetMs: endpointingPreset("balanced").deepgramEndpointing,
    vadMs: null,
    sttMs: null,
    toolMs: null,
    llmMs: null,
    ttsFirstMs: null,
    totalMs: null,
    voiceMs: null,
    sequentialMs: null,
    overlapSavedMs: null,
  };
}

function freshBargeIn() {
  return {
    attempts: 0,
    successes: 0,
    active: false,
    startedAt: 0,
    stoppedAt: 0,
    capturedAt: 0,
    stopLatencyMs: null,
    captureLatencyMs: null,
    replacementText: "",
    status: "No interruption measured yet.",
  };
}

function defaultToolServerUrl() {
  if (import.meta.env.VITE_VAPI_TOOL_SERVER_URL) return import.meta.env.VITE_VAPI_TOOL_SERVER_URL;
  if (["localhost", "127.0.0.1"].includes(location.hostname) && location.port !== "8000") {
    return "http://localhost:8000/api/vapi/webhook";
  }
  return `${location.origin}/api/vapi/webhook`;
}

function backendUrl(path) {
  try {
    const webhook = new URL(els.toolServerUrl.value.trim());
    return new URL(path, webhook.origin).href;
  } catch {
    return path;
  }
}

function previewBackendUrl(path) {
  if (["localhost", "127.0.0.1"].includes(location.hostname) && location.port === "8000") {
    return new URL(path, location.origin).href;
  }
  return backendUrl(path);
}

function restoreConfig() {
  els.vapiPublicKey.value = import.meta.env.VITE_VAPI_PUBLIC_KEY || localStorage.getItem("aislepilot.vapiPublicKey") || "";
  els.toolServerUrl.value = localStorage.getItem("aislepilot.toolServerUrl") || defaultToolServerUrl();
  els.voiceId.value = import.meta.env.VITE_ELEVENLABS_VOICE_ID || localStorage.getItem("aislepilot.voiceId") || "burt";
  state.voiceAssistantId = import.meta.env.VITE_VAPI_ASSISTANT_ID || localStorage.getItem("aislepilot.voiceAssistantId") || "";
  setEndpointingMode(localStorage.getItem("aislepilot.endpointingMode") || "balanced", { persist: false });
}

function applyVoiceConfig(config = {}, { persist = true } = {}) {
  const publicKey = String(config.voiceAgentPublicKey || config.agentKey || config.publicKey || "").trim();
  const assistantId = String(config.voiceAssistantId || config.assistantId || "").trim();
  const serviceUrl = String(config.serviceUrl || config.toolServerUrl || "").trim();
  const voiceId = String(config.voiceProfileId || config.voiceId || "").trim();
  if (publicKey) els.vapiPublicKey.value = publicKey;
  if (assistantId) state.voiceAssistantId = assistantId;
  if (serviceUrl) els.toolServerUrl.value = serviceUrl;
  if (voiceId) els.voiceId.value = voiceId;
  if (persist && (publicKey || assistantId || serviceUrl || voiceId)) saveConfig();
}

function configFromUrl() {
  const search = new URLSearchParams(window.location.search);
  const hash = new URLSearchParams(window.location.hash.replace(/^#/, ""));
  const get = (...names) => {
    for (const name of names) {
      const value = hash.get(name) || search.get(name);
      if (value) return value;
    }
    return "";
  };
  return {
    voiceAgentPublicKey: get("agent_key", "voice_key", "public_key"),
    voiceAssistantId: get("assistant_id", "voice_assistant_id"),
    voiceProfileId: get("voice_id", "voice_profile"),
    serviceUrl: get("service_url", "tool_url"),
  };
}

function allowUrlVoiceConfig() {
  return ["localhost", "127.0.0.1"].includes(location.hostname);
}

function publicConfigUrls() {
  const urls = [new URL("/api/public-config", location.origin).href];
  const fallback = previewBackendUrl("/api/public-config");
  if (!urls.includes(fallback)) urls.push(fallback);
  return urls;
}

async function fetchPublicConfig(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error("Public voice config unavailable");
  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) throw new Error("Public voice config is not JSON");
  return response.json();
}

async function loadPublicVoiceConfig({ force = false } = {}) {
  if (publicConfigLoad && !force) return publicConfigLoad;
  publicConfigLoad = (async () => {
    const urlConfig = allowUrlVoiceConfig() ? configFromUrl() : {};
    applyVoiceConfig(urlConfig, { persist: true });
    for (const url of publicConfigUrls()) {
      try {
        const config = await fetchPublicConfig(url);
        applyVoiceConfig(config, { persist: true });
        applyVoiceConfig(urlConfig, { persist: true });
        return config;
      } catch {
        // Try the next known backend origin. The call card handles missing config gracefully.
      }
    }
    return null;
  })();
  return publicConfigLoad;
}

function saveConfig() {
  localStorage.setItem("aislepilot.vapiPublicKey", els.vapiPublicKey.value.trim());
  localStorage.setItem("aislepilot.voiceAssistantId", state.voiceAssistantId.trim());
  localStorage.setItem("aislepilot.toolServerUrl", els.toolServerUrl.value.trim());
  localStorage.setItem("aislepilot.voiceId", els.voiceId.value.trim());
  localStorage.setItem("aislepilot.endpointingMode", state.endpointingMode);
}

function setEndpointingMode(mode, { persist = true } = {}) {
  state.endpointingMode = endpointingPreset(mode) === endpointingPreset("balanced") && mode !== "balanced" ? "balanced" : mode;
  if (!endpointingPreset(state.endpointingMode)) state.endpointingMode = "balanced";
  if (persist) saveConfig();
  stampEndpointingMetrics();
  renderEndpointingControls();
  renderSpeechBenchmark();
}

function stampEndpointingMetrics() {
  const preset = endpointingPreset(state.endpointingMode);
  state.metrics.endpointingMode = state.endpointingMode;
  state.metrics.endpointingTargetMs = preset.deepgramEndpointing;
}

function renderEndpointingControls() {
  const preset = endpointingPreset(state.endpointingMode);
  els.endpointingButtons.forEach((button) => {
    const active = button.dataset.endpointingMode === state.endpointingMode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  if (els.endpointingPresetSummary) {
    els.endpointingPresetSummary.textContent = `${preset.label}: ${preset.deepgramEndpointing} ms speech endpoint, ${preset.silero.redemptionMs} ms voice release.`;
  }
  if (els.endpointingModeMetric) {
    els.endpointingModeMetric.textContent = preset.label;
  }
}

function compactId(value) {
  if (!value) return "none";
  const text = String(value);
  return text.length > 18 ? `${text.slice(0, 12)}...` : text;
}

function traceId() {
  const stamp = new Date().toISOString().replace(/[-:T.Z]/g, "").slice(0, 14);
  const suffix = Math.random().toString(16).slice(2, 8);
  return `trace-${stamp}-${suffix}`;
}

function beginTrace(source = "live_voice") {
  state.trace = {
    traceId: traceId(),
    source,
    architecture: "vapi-managed-cascade",
    status: "recording",
    createdAt: new Date().toISOString(),
    startedAtPerf: performance.now(),
    pipeline: {
      orchestrator: "vapi",
      vad: "silero-v5-browser",
      stt: "deepgram-nova-3",
      llm: "openai-gpt-4o-mini",
      rag: "advanced-hybrid-rrf-rerank",
      tts: "elevenlabs-turbo-v2.5",
    },
    events: [],
    metrics: {},
    turnCount: 0,
  };
  state.latestReplay = null;
  state.latestEvaluation = null;
  els.traceReplayRows.textContent = "Recording current session.";
  els.traceTrustRows.textContent = "Recording current session.";
  recordTraceEvent("trace_start", { source });
  updateTracePanel();
}

function traceRelativeMs() {
  return state.trace ? Math.round(performance.now() - state.trace.startedAtPerf) : 0;
}

function recordTraceEvent(type, payload = {}) {
  if (!state.trace) return;
  state.trace.events.push({
    type,
    relativeMs: traceRelativeMs(),
    at: new Date().toISOString(),
    payload: JSON.parse(JSON.stringify(payload ?? {})),
  });
  updateTracePanel();
}

function durationMs(start, end) {
  if (!start || !end || end < start) return null;
  return Math.round(end - start);
}

function updateLatencyDerived() {
  const metrics = state.metrics;
  metrics.sttMs = durationMs(metrics.speechEndedAt, metrics.userFinalAt) ?? metrics.sttMs;
  metrics.toolMs = durationMs(metrics.toolStartedAt, metrics.toolEndedAt) ?? metrics.toolMs;
  metrics.llmMs = durationMs(metrics.toolEndedAt || metrics.userFinalAt || metrics.speechEndedAt, metrics.assistantFirstAt) ?? metrics.llmMs;
  metrics.ttsFirstMs = durationMs(metrics.assistantFirstAt, metrics.voiceStartedAt) ?? metrics.ttsFirstMs;
  metrics.voiceMs = durationMs(metrics.speechEndedAt, metrics.voiceStartedAt) ?? metrics.voiceMs;
  metrics.totalMs = metrics.voiceMs ?? metrics.totalMs;

  const sequentialParts = [metrics.sttMs, metrics.toolMs, metrics.llmMs, metrics.ttsFirstMs].filter(Number.isFinite);
  metrics.sequentialMs = sequentialParts.length ? sequentialParts.reduce((sum, value) => sum + value, 0) : null;
  metrics.overlapSavedMs =
    Number.isFinite(metrics.sequentialMs) && Number.isFinite(metrics.voiceMs)
      ? Math.max(0, metrics.sequentialMs - metrics.voiceMs)
      : null;
}

function beginBargeIn(source, text = "") {
  if (!state.callActive) return;
  const now = performance.now();
  if (!state.assistantSpeaking && !state.bargeIn.active) return;

  if (!state.bargeIn.active) {
    state.bargeIn = {
      ...state.bargeIn,
      attempts: state.bargeIn.attempts + 1,
      active: true,
      startedAt: now,
      stoppedAt: 0,
      capturedAt: 0,
      stopLatencyMs: null,
      captureLatencyMs: null,
      replacementText: "",
      status: "Interrupt detected. Waiting for assistant audio to stop.",
    };
    recordTraceEvent("barge_in_start", { source, turn: state.metrics.turn || state.completedTurn + 1 });
  }

  if (text) markBargeInCaptured(text, source);
  renderSpeechBenchmark();
}

function markBargeInStopped(source) {
  if (!state.bargeIn.active || state.bargeIn.stoppedAt) return;
  const now = performance.now();
  state.bargeIn.stoppedAt = now;
  state.bargeIn.stopLatencyMs = durationMs(state.bargeIn.startedAt, now);
  state.bargeIn.successes += 1;
  state.bargeIn.status = "Assistant audio stopped after interruption.";
  recordTraceEvent("barge_in_stop", {
    source,
    stopLatencyMs: state.bargeIn.stopLatencyMs,
    turn: state.metrics.turn || state.completedTurn + 1,
  });
  if (state.bargeIn.capturedAt) state.bargeIn.active = false;
  renderSpeechBenchmark();
}

function markBargeInCaptured(text, source) {
  if (!state.bargeIn.active || state.bargeIn.capturedAt) return;
  const now = performance.now();
  state.bargeIn.capturedAt = now;
  state.bargeIn.captureLatencyMs = durationMs(state.bargeIn.startedAt, now);
  state.bargeIn.replacementText = text;
  state.bargeIn.status = `Replacement query captured: "${text}"`;
  recordTraceEvent("barge_in_capture", {
    source,
    captureLatencyMs: state.bargeIn.captureLatencyMs,
    transcriptPreview: text.slice(0, 160),
  });
  renderSpeechBenchmark();
}

function updateTracePanel() {
  const trace = state.trace;
  const events = trace?.events || [];
  const toolCalls = events.filter((event) => event.type === "tool_call").length;
  const liveRow = state.latestLedger?.rows?.find((row) => row.id === "vapi-stack");
  els.traceStatus.textContent = trace?.status || "Idle";
  els.traceCount.textContent = compactId(trace?.traceId);
  els.traceEventCount.textContent = String(events.length);
  els.traceToolCount.textContent = String(toolCalls);
  const replayRate = state.latestReplay?.replay?.deterministicMatchRate;
  els.traceReplayScore.textContent = replayRate === null || replayRate === undefined ? "-" : `${Math.round(replayRate * 100)}%`;
  els.traceTrustScore.textContent = state.latestEvaluation?.trustScore === undefined ? "-" : `${state.latestEvaluation.trustScore}`;
  els.traceCost.textContent = formatMoney(liveRow?.cost || 0);
}

function traceForSave(status = "completed") {
  if (!state.trace) return null;
  const { startedAtPerf, ...trace } = state.trace;
  return {
    ...trace,
    status,
    endedAt: new Date().toISOString(),
    durationMs: traceRelativeMs(),
    turnCount: state.turnCount,
    metrics: {
      ...state.metrics,
      totalTurns: state.turnCount,
      bargeIn: state.bargeIn,
    },
    costEstimate: state.latestLedger,
  };
}

async function saveCurrentTrace(status = "completed") {
  const trace = traceForSave(status);
  if (!trace) return;
  state.trace.status = "saving";
  updateTracePanel();
  try {
    const response = await fetch(previewBackendUrl("/api/traces"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(trace),
    });
    if (!response.ok) throw new Error("Trace save failed");
    const saved = await response.json();
    state.trace.status = "saved";
    const savedId = saved.summary?.traceId || trace.traceId;
    els.traceCount.textContent = compactId(savedId);
    await loadTraces();
    try {
      await evaluateTrace(savedId);
    } catch {
      els.traceTrustRows.innerHTML = `<div class="trace-empty">Trace saved, but trust evaluation failed.</div>`;
    }
  } catch {
    state.trace.status = "save failed";
  }
  updateTracePanel();
}

async function loadTraces() {
  try {
    const response = await fetch(previewBackendUrl("/api/traces?limit=8"));
    if (!response.ok) throw new Error("Trace list unavailable");
    const payload = await response.json();
    renderTraceList(payload.traces || []);
  } catch {
    els.traceList.innerHTML = `<div class="trace-empty">Trace service unavailable.</div>`;
  }
}

function renderTraceList(traces) {
  if (!traces.length) {
    els.traceList.innerHTML = `<div class="trace-empty">No saved traces yet.</div>`;
    return;
  }
  els.traceList.replaceChildren(
    ...traces.map((trace) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "trace-item";
      const title = document.createElement("strong");
      title.textContent = compactId(trace.traceId);
      const meta = document.createElement("span");
      meta.textContent = `${trace.turns || 0} turns / ${trace.toolCalls || 0} tools / ${formatMoney(trace.estimatedCost || 0)}`;
      const status = document.createElement("span");
      status.textContent = `${trace.status || "recorded"} / ${formatMs(trace.latencyMs)} / ${trace.savedAt || trace.createdAt || ""}`;
      button.append(title, meta, status);
      button.addEventListener("click", () => replayTrace(trace.traceId));
      return button;
    }),
  );
}

async function replayTrace(id) {
  els.traceReplayRows.textContent = "Replaying trace...";
  els.traceTrustRows.textContent = "Evaluating trace...";
  try {
    const response = await fetch(previewBackendUrl(`/api/traces/${encodeURIComponent(id)}/replay`), {
      method: "POST",
    });
    if (!response.ok) throw new Error("Trace replay failed");
    state.latestReplay = await response.json();
    renderReplay(state.latestReplay);
    try {
      await evaluateTrace(id);
    } catch {
      els.traceTrustRows.innerHTML = `<div class="trace-empty">Trust evaluation failed for this trace.</div>`;
    }
  } catch {
    els.traceReplayRows.innerHTML = `<div class="trace-empty">Replay failed for this trace.</div>`;
    els.traceTrustRows.innerHTML = `<div class="trace-empty">Trust evaluation skipped because replay failed.</div>`;
  }
  updateTracePanel();
}

async function evaluateTrace(id) {
  const response = await fetch(previewBackendUrl(`/api/evaluation/traces/${encodeURIComponent(id)}`));
  if (!response.ok) throw new Error("Trace evaluation failed");
  state.latestEvaluation = await response.json();
  renderEvaluation(state.latestEvaluation);
}

function renderReplay(payload) {
  const items = payload?.replay?.items || [];
  const rate = payload?.replay?.deterministicMatchRate;
  els.traceReplayScore.textContent = rate === null || rate === undefined ? "-" : `${Math.round(rate * 100)}%`;
  if (!items.length) {
    els.traceReplayRows.innerHTML = `<div class="trace-empty">No tool calls to replay.</div>`;
    return;
  }
  els.traceReplayRows.replaceChildren(
    ...items.map((item) => {
      const row = document.createElement("div");
      row.className = `trace-replay-row ${item.deterministicMatch ? "pass" : "review"}`;
      const title = document.createElement("strong");
      title.textContent = `${item.deterministicMatch ? "PASS" : "REVIEW"} - ${item.tool}`;
      const meta = document.createElement("span");
      meta.textContent = `${item.query} / stored ${item.storedFound} / current ${item.currentFound}`;
      const detail = document.createElement("span");
      const current = item.currentSources?.length ? item.currentSources.join(", ") : item.currentItem || "-";
      detail.textContent = `Current evidence: ${current}`;
      row.append(title, meta, detail);
      return row;
    }),
  );
}

function renderEvaluation(payload) {
  const findings = payload?.findings || [];
  els.traceTrustScore.textContent = payload?.trustScore === undefined ? "-" : `${payload.trustScore}`;
  if (!findings.length) {
    els.traceTrustRows.innerHTML = `<div class="trace-empty">No trust findings.</div>`;
    return;
  }
  els.traceTrustRows.replaceChildren(
    ...findings.map((finding) => {
      const row = document.createElement("div");
      row.className = `trace-replay-row ${finding.severity === "info" ? "pass" : "review"}`;
      const title = document.createElement("strong");
      title.textContent = `${String(finding.severity || "info").toUpperCase()} - ${finding.label}`;
      const detail = document.createElement("span");
      detail.textContent = finding.detail || "";
      row.append(title, detail);
      return row;
    }),
  );
}

function setConnection(status, text) {
  els.connectionDot.classList.toggle("online", status === "online");
  els.connectionDot.classList.toggle("recording", status === "recording");
  els.connectionText.textContent = text;
  updateSimpleStatus(status, text);
}

function setVad(status, text) {
  els.vadDot.classList.toggle("online", status === "ready");
  els.vadDot.classList.toggle("recording", status === "speech");
  els.vadText.textContent = text;
  if (status === "speech") updateSimpleStatus("recording", "Listening");
}

function setStage(stage, label) {
  els.activeStage.textContent = label || stage || "Idle";
  document.querySelectorAll(".node").forEach((node) => {
    node.classList.toggle("active", node.dataset.stage === stage);
  });
  if (stage === "tts") updateSimpleStatus("online", "Speaking");
  if (stage === "llm" || stage === "stt") updateSimpleStatus("online", "Thinking");
  if (stage === "vad" && state.callActive) updateSimpleStatus("online", "Listening");
}

function setCallUi(active, pending = false) {
  const label = pending ? "Starting..." : active ? "End voice session" : "Start voice session";
  els.callLabel.textContent = label;
  els.callButtonTop.textContent = pending ? "Starting" : active ? "End" : "Start";
  els.callButton.classList.toggle("recording", active);
  els.callButton.setAttribute("aria-pressed", String(active));
  els.callButton.disabled = pending;
  els.callButtonTop.disabled = pending;
  [els.vapiPublicKey, els.toolServerUrl, els.voiceId].forEach((input) => {
    input.disabled = active || pending;
  });
  els.endpointingButtons.forEach((button) => {
    button.disabled = active || pending;
  });
  els.simpleCallButton?.classList.toggle("active", active || pending);
  els.simpleCallButton?.classList.toggle("pending", pending);
  if (els.simpleHint) {
    els.simpleHint.textContent = pending
      ? "Connecting the shopping assistant..."
      : active
        ? "Ask a product, stock, recommendation, or store-policy question."
        : "Tap the center button and ask a shopping question.";
  }
  if (els.simpleEndButton) els.simpleEndButton.disabled = pending;
  if (els.simpleEndButton) els.simpleEndButton.querySelector("span").textContent = active || pending ? "End Call" : "Close";
}

async function toggleCall() {
  if (state.starting) return;
  if (state.callActive) {
    await endCall();
  } else {
    await startCall();
  }
}

async function vapiStartPayload({ toolServerUrl, voiceId }) {
  if (state.voiceAssistantId.trim()) return state.voiceAssistantId.trim();
  const { buildAssistant } = await import("./assistant-config.js");
  return buildAssistant({ toolServerUrl, voiceId, endpointingMode: state.endpointingMode });
}

async function startCall() {
  hideFeedbackPanel();
  clearIdleTimers();
  state.callEndReason = "manual";
  await loadPublicVoiceConfig({ force: true });
  const publicKey = els.vapiPublicKey.value.trim();
  const assistantId = state.voiceAssistantId.trim();
  const toolServerUrl = els.toolServerUrl.value.trim();
  const voiceId = els.voiceId.value.trim();

  if (!publicKey) {
    els.answerText.textContent = VOICE_UNAVAILABLE_TEXT;
    if (els.simpleSetupMessage) els.simpleSetupMessage.textContent = VOICE_UNAVAILABLE_TEXT;
    return;
  }

  let toolServer;
  try {
    toolServer = new URL(toolServerUrl);
  } catch {
    els.answerText.textContent = VOICE_UNAVAILABLE_TEXT;
    if (els.simpleSetupMessage) els.simpleSetupMessage.textContent = VOICE_UNAVAILABLE_TEXT;
    return;
  }

  if (toolServer.protocol !== "https:" || ["localhost", "127.0.0.1", "[::1]"].includes(toolServer.hostname)) {
    els.answerText.textContent = VOICE_UNAVAILABLE_TEXT;
    if (els.simpleSetupMessage) els.simpleSetupMessage.textContent = VOICE_UNAVAILABLE_TEXT;
    return;
  }

  if (!assistantId && !voiceId) {
    els.answerText.textContent = VOICE_UNAVAILABLE_TEXT;
    if (els.simpleSetupMessage) els.simpleSetupMessage.textContent = VOICE_UNAVAILABLE_TEXT;
    return;
  }

  saveConfig();
  beginTrace("live_voice");
  recordTraceEvent("config", {
    toolServerOrigin: toolServer.origin,
    voiceProvider: "elevenlabs",
    orchestrator: "vapi",
    endpointingMode: state.endpointingMode,
    endpointingPreset: endpointingPreset(state.endpointingMode),
  });
  state.starting = true;
  state.metrics = freshMetrics();
  stampEndpointingMetrics();
  state.bargeIn = freshBargeIn();
  state.lastAssistantSpeechStartAt = 0;
  state.lastAssistantSpeechEndAt = 0;
  state.userText = "";
  state.answer = "";
  state.callEndedAt = 0;
  state.lastCallSeconds = 0;
  state.turnCount = 0;
  state.completedTurn = 0;
  state.idlePrompted = false;
  state.lastActivityAt = performance.now();
  els.turnCount.textContent = "0 turns";
  renderMetrics();
  setCallUi(false, true);
  setConnection("offline", "Connecting");
  setStage("vad", "Preparing call");

  try {
    state.vapi = new Vapi(publicKey);
    bindVapiEvents(state.vapi);
    await state.vapi.start(await vapiStartPayload({ toolServerUrl, voiceId }));
  } catch (error) {
    recordTraceEvent("error", { stage: "start_call", message: readableError(error) });
    state.starting = false;
    setCallUi(false);
    setConnection("offline", "Start failed");
    els.answerText.textContent = readableError(error);
    if (els.simpleAnswer) els.simpleAnswer.textContent = VOICE_UNAVAILABLE_TEXT;
    if (els.simpleSetupMessage) els.simpleSetupMessage.textContent = VOICE_UNAVAILABLE_TEXT;
  }
}

async function endCall(reason = "manual") {
  state.callEndReason = reason;
  setConnection("offline", "Ending");
  try {
    if (state.vapi) await state.vapi.stop();
  } finally {
    await finalizeCall();
  }
}

function bindVapiEvents(vapi) {
  vapi.on("call-start-success", () => activateCall(vapi));
  vapi.on("call-start", () => activateCall(vapi));
  vapi.on("call-start-progress", (event) => {
    recordTraceEvent("call_start_progress", { stage: event?.stage });
    if (!state.callActive && event?.stage) {
      setConnection("offline", `Starting: ${formatStageName(event.stage)}`);
    }
  });
  vapi.on("call-start-failed", (event) => {
    recordTraceEvent("error", { stage: "call_start_failed", message: event?.error || "Voice call could not start." });
    state.starting = false;
    setCallUi(false);
    setConnection("offline", "Start failed");
    setStage("idle", "Idle");
    els.answerText.textContent = event?.error || "Voice call could not start.";
    if (els.simpleAnswer) els.simpleAnswer.textContent = VOICE_UNAVAILABLE_TEXT;
    if (els.simpleSetupMessage) els.simpleSetupMessage.textContent = VOICE_UNAVAILABLE_TEXT;
  });

  vapi.on("call-end", () => {
    recordTraceEvent("call_end", {});
    finalizeCall();
  });

  vapi.on("speech-start", async () => {
    if (state.idlePrompted) {
      clearIdlePromptTimer();
    } else {
      clearIdleTimers();
    }
    recordTraceEvent("assistant_speech_start", {});
    state.assistantSpeaking = true;
    state.lastAssistantSpeechStartAt = performance.now();
    state.metrics.voiceStartedAt = performance.now();
    renderMetrics();
    setStage("tts", "Speaking");
    state.vapi?.setMuted(false);
    if (state.vad?.listening) await state.vad.pause();
    setVad("ready", "Barge-in listening");
  });

  vapi.on("speech-end", async () => {
    recordTraceEvent("assistant_speech_end", {});
    state.assistantSpeaking = false;
    state.lastAssistantSpeechEndAt = performance.now();
    state.metrics.assistantSpeechEndedAt = state.lastAssistantSpeechEndAt;
    const wasBargeIn = state.bargeIn.active;
    markBargeInStopped("assistant_speech_end");
    if (!wasBargeIn) completeLedgerTurn();
    state.vapi?.setMuted(false);
    if (state.callActive && state.vad && !state.vad.listening) await state.vad.start();
    setVad("ready", "Listening");
    setStage("vad", "Listening");
    markCallActivity("assistant_speech_end", { resetPrompt: false });
    if (state.idlePrompted) {
      if (state.idleEndTimer) window.clearTimeout(state.idleEndTimer);
      state.idleEndTimer = window.setTimeout(() => endForIdle(), IDLE_END_AFTER_PROMPT_MS);
      if (els.simpleHint) els.simpleHint.textContent = "Waiting for your response...";
      return;
    }
    scheduleIdleFollowup();
  });

  vapi.on("message", handleVapiMessage);
  vapi.on("error", (error) => {
    recordTraceEvent("error", { stage: "vapi", message: readableError(error) });
    if (state.starting) {
      state.starting = false;
      setCallUi(false);
      setStage("idle", "Idle");
    }
    els.answerText.textContent = readableError(error);
    if (els.simpleAnswer) els.simpleAnswer.textContent = VOICE_UNAVAILABLE_TEXT;
    if (els.simpleSetupMessage) els.simpleSetupMessage.textContent = VOICE_UNAVAILABLE_TEXT;
    setConnection("offline", "Voice error");
  });
}

async function activateCall(vapi) {
  if (state.callActive) return;
  state.starting = false;
  state.callActive = true;
  state.callStartedAt = performance.now();
  state.lastActivityAt = state.callStartedAt;
  startSimpleTimer();
  recordTraceEvent("call_start", { connected: true });
  setCallUi(true);
  setConnection("online", "Live");
  els.answerText.textContent = "Connected. Ask a shopping question.";
  if (els.simpleAnswer) els.simpleAnswer.textContent = "Connected. Ask me where an item is, what is in stock, or which option is best.";
  refreshLedger();
  state.ledgerTimer = window.setInterval(() => {
    if (state.callActive) refreshLedger();
  }, 2500);

  try {
    await initializeVad();
    if (!state.assistantSpeaking) {
      vapi.setMuted(false);
      await state.vad.start();
      setVad("ready", "Listening");
      setStage("vad", "Listening");
      scheduleIdleFollowup();
    } else {
      vapi.setMuted(false);
      setVad("ready", "Barge-in listening");
    }
  } catch (error) {
    setVad("error", "Voice listener failed");
    els.answerText.textContent = `${readableError(error)} The mic is still open; try asking a product question.`;
    if (els.simpleAnswer) els.simpleAnswer.textContent = "The voice listener had an issue, but the call may still be open. Try asking a product question.";
    vapi.setMuted(false);
  }
}

function formatStageName(stage) {
  return String(stage).replaceAll("-", " ");
}

function formatClock(ms) {
  const totalSeconds = Math.max(0, Math.floor(Number(ms || 0) / 1000));
  const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function updateSimpleTimer() {
  if (!els.simpleTimer) return;
  els.simpleTimer.textContent = state.callStartedAt ? formatClock(performance.now() - state.callStartedAt) : "00:00";
}

function startSimpleTimer() {
  if (state.simpleTimer) window.clearInterval(state.simpleTimer);
  updateSimpleTimer();
  state.simpleTimer = window.setInterval(updateSimpleTimer, 1000);
}

function stopSimpleTimer() {
  if (state.simpleTimer) window.clearInterval(state.simpleTimer);
  state.simpleTimer = null;
  updateSimpleTimer();
}

function clearIdleTimers() {
  if (state.idlePromptTimer) window.clearTimeout(state.idlePromptTimer);
  if (state.idleEndTimer) window.clearTimeout(state.idleEndTimer);
  state.idlePromptTimer = null;
  state.idleEndTimer = null;
}

function clearIdlePromptTimer() {
  if (state.idlePromptTimer) window.clearTimeout(state.idlePromptTimer);
  state.idlePromptTimer = null;
}

function markCallActivity(source = "activity", { resetPrompt = true } = {}) {
  state.lastActivityAt = performance.now();
  if (resetPrompt) state.idlePrompted = false;
  recordTraceEvent("call_activity", { source, resetPrompt });
}

function scheduleIdleFollowup() {
  clearIdleTimers();
  if (!state.callActive || state.starting || state.assistantSpeaking) return;
  state.idlePromptTimer = window.setTimeout(() => promptIfIdle(), IDLE_PROMPT_AFTER_MS);
}

async function promptIfIdle() {
  state.idlePromptTimer = null;
  if (!state.callActive || state.starting || state.assistantSpeaking || state.idlePrompted) return;
  state.idlePrompted = true;
  recordTraceEvent("idle_followup_prompt", {
    idleMs: Math.round(performance.now() - (state.lastActivityAt || state.callStartedAt || performance.now())),
  });
  if (els.simpleHint) els.simpleHint.textContent = "Checking if you still need help...";
  if (els.simpleAnswer) els.simpleAnswer.textContent = "Do you need any more help?";
  await speakAssistantPrompt("Do you need any more help?");
  state.idleEndTimer = window.setTimeout(() => endForIdle(), IDLE_END_AFTER_PROMPT_MS + 3500);
}

async function speakAssistantPrompt(text) {
  if (!state.vapi) return;
  try {
    if (typeof state.vapi.say === "function") {
      state.vapi.say(text, false);
      return;
    }
    state.vapi.send({
      type: "add-message",
      message: {
        role: "user",
        content: `The shopper has been silent. Ask exactly this and nothing else: "${text}"`,
      },
      triggerResponseEnabled: true,
    });
  } catch (error) {
    recordTraceEvent("idle_followup_error", { message: readableError(error) });
  }
}

async function endForIdle() {
  state.idleEndTimer = null;
  if (!state.callActive || state.starting || state.assistantSpeaking) return;
  recordTraceEvent("idle_auto_end", {
    idleMs: Math.round(performance.now() - (state.lastActivityAt || state.callStartedAt || performance.now())),
  });
  state.callEndReason = "idle_timeout";
  if (els.simpleAnswer) els.simpleAnswer.textContent = "No response detected. Ending the call now.";
  await endCall("idle_timeout");
}

function updateSimpleStatus(status, text) {
  if (!els.simpleStatusText || !els.simpleStatusDot) return;
  const publicText = {
    "Connecting to Vapi": "Connecting",
    "Vapi connected": "Live",
    "Vapi error": "Unavailable",
    "Needs setup": "Unavailable",
    Offline: "Ready",
    Processing: "Thinking",
    "Customer speaking": "Listening",
    "Interrupting assistant": "Listening",
  }[text] || text || "Ready";
  els.simpleStatusText.textContent = publicText;
  els.simpleStatusDot.classList.toggle("live", status === "online" || status === "recording");
  els.simpleStatusDot.classList.toggle("recording", status === "recording");
}

function voiceConfigStatus() {
  const publicKey = els.vapiPublicKey?.value.trim();
  const assistantId = state.voiceAssistantId.trim();
  const toolServerUrl = els.toolServerUrl?.value.trim();
  const voiceId = els.voiceId?.value.trim();
  if (!publicKey || (!assistantId && !voiceId) || !toolServerUrl) return { ready: false, reason: VOICE_UNAVAILABLE_TEXT };
  try {
    const toolServer = new URL(toolServerUrl);
    const local = ["localhost", "127.0.0.1", "[::1]"].includes(toolServer.hostname);
    if (toolServer.protocol !== "https:" || local) {
      return { ready: false, reason: VOICE_UNAVAILABLE_TEXT };
    }
  } catch {
    return { ready: false, reason: VOICE_UNAVAILABLE_TEXT };
  }
  return { ready: true, reason: "" };
}

async function openSimpleAgent({ autoStart = true } = {}) {
  document.body.classList.add("voice-call-open");
  els.voiceCallModal?.setAttribute("aria-hidden", "false");
  if (els.simpleSetupMessage) els.simpleSetupMessage.textContent = "";
  if (!state.callActive && !state.starting) {
    els.simpleTranscript.textContent = "No question yet.";
    els.simpleAnswer.textContent = "Ready to help.";
  }
  updateSimpleStatus(state.callActive ? "online" : "offline", state.callActive ? "Live" : "Ready");
  updateSimpleTimer();
  if (!autoStart || state.callActive || state.starting) return;
  await loadPublicVoiceConfig({ force: true });
  const config = voiceConfigStatus();
  if (!config.ready) {
    if (els.simpleSetupMessage) els.simpleSetupMessage.textContent = config.reason;
    updateSimpleStatus("offline", "Unavailable");
    return;
  }
  startCall();
}

function closeSimpleAgent() {
  document.body.classList.remove("voice-call-open");
  els.voiceCallModal?.setAttribute("aria-hidden", "true");
}

function hideFeedbackPanel() {
  state.feedbackScore = null;
  if (els.feedbackPanel) els.feedbackPanel.hidden = true;
  if (els.feedbackThanks) els.feedbackThanks.textContent = "";
  if (els.feedbackComment) els.feedbackComment.value = "";
  els.feedbackButtons.forEach((button) => button.classList.remove("selected"));
}

function showFeedbackPanel() {
  if (!els.feedbackPanel) return;
  els.feedbackPanel.hidden = false;
  if (els.feedbackThanks) {
    els.feedbackThanks.textContent =
      state.callEndReason === "idle_timeout"
        ? "Call ended after no response. Please rate the help quality."
        : "Please rate this call.";
  }
}

function selectFeedbackScore(score) {
  state.feedbackScore = Number(score);
  els.feedbackButtons.forEach((button) => {
    button.classList.toggle("selected", Number(button.dataset.feedbackScore) === state.feedbackScore);
  });
  if (els.feedbackThanks) els.feedbackThanks.textContent = "";
}

function localFeedbackSummary(payload) {
  const key = "aislepilot.feedbackSummary";
  const current = JSON.parse(localStorage.getItem(key) || "{\"count\":0,\"happy\":0,\"scores\":[]}");
  const score = Number(payload.score || 0);
  const next = {
    count: Number(current.count || 0) + 1,
    happy: Number(current.happy || 0) + (score >= 4 ? 1 : 0),
    scores: [...(Array.isArray(current.scores) ? current.scores : []), score].slice(-100),
  };
  localStorage.setItem(key, JSON.stringify(next));
  return {
    count: next.count,
    happy: next.happy,
    happyRate: next.count ? next.happy / next.count : 0,
  };
}

async function submitFeedback() {
  if (!state.feedbackScore) {
    if (els.feedbackThanks) els.feedbackThanks.textContent = "Choose a rating from 1 to 5.";
    return;
  }
  const payload = {
    score: state.feedbackScore,
    happy: state.feedbackScore >= 4,
    comment: els.feedbackComment?.value.trim() || "",
    endReason: state.callEndReason,
    callSeconds: state.lastCallSeconds,
    turnCount: state.turnCount,
    userText: state.userText,
    answer: state.answer,
    traceId: state.trace?.traceId || null,
    createdAt: new Date().toISOString(),
  };
  state.lastFeedbackPayload = payload;
  if (els.feedbackSubmitButton) els.feedbackSubmitButton.disabled = true;
  try {
    const response = await fetch(previewBackendUrl("/api/feedback"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error("feedback unavailable");
    const saved = await response.json();
    const summary = saved.summary || {};
    const happyRate = Number(summary.happyRate || 0);
    if (els.feedbackThanks) {
      els.feedbackThanks.textContent = `Thanks. Saved ${summary.total || 1} feedback entries, happy rate ${formatPercent(happyRate)}.`;
    }
    renderFeedbackSummary(summary);
  } catch {
    const summary = localFeedbackSummary(payload);
    if (els.feedbackThanks) {
      els.feedbackThanks.textContent = `Thanks. Saved locally: ${summary.count} entries, happy rate ${formatPercent(summary.happyRate)}.`;
    }
    renderFeedbackSummary({ total: summary.count, happy: summary.happy, happyRate: summary.happyRate });
  } finally {
    if (els.feedbackSubmitButton) els.feedbackSubmitButton.disabled = false;
  }
}

function renderFeedbackSummary(summary) {
  if (!els.feedbackHappyMetric) return;
  if (!summary || !Number(summary.total || 0)) {
    els.feedbackHappyMetric.textContent = "-";
    return;
  }
  els.feedbackHappyMetric.textContent = `${summary.happy || 0}/${summary.total} (${formatPercent(summary.happyRate || 0)})`;
}

async function loadFeedbackSummary() {
  if (!els.feedbackHappyMetric) return;
  try {
    const response = await fetch(previewBackendUrl("/api/feedback/summary"));
    if (!response.ok) throw new Error("feedback summary unavailable");
    const payload = await response.json();
    renderFeedbackSummary(payload.summary);
  } catch {
    const summary = JSON.parse(localStorage.getItem("aislepilot.feedbackSummary") || "null");
    if (summary) renderFeedbackSummary({ total: summary.count, happy: summary.happy, happyRate: summary.count ? summary.happy / summary.count : 0 });
  }
}

async function toggleSimpleCall() {
  if (state.callActive || state.starting) {
    await endCall();
    return;
  }
  await loadPublicVoiceConfig({ force: true });
  const config = voiceConfigStatus();
  if (!config.ready) {
    if (els.simpleSetupMessage) els.simpleSetupMessage.textContent = config.reason;
    updateSimpleStatus("offline", "Unavailable");
    return;
  }
  await startCall();
}

function openAgentWorkspace(target = "voice") {
  document.body.classList.add("agent-open");
  els.agentDock?.setAttribute("aria-hidden", "false");
  const targetSelector =
    target === "benchmarks"
      ? ".benchmark-panel"
      : target === "ledger"
        ? ".ledger-panel"
        : ".voice-panel";
  window.setTimeout(() => {
    if (target === "voice") {
      els.agentDockPanel?.scrollTo({ top: 0, behavior: "smooth" });
    } else {
      const panel = document.querySelector(targetSelector) || els.callButton;
      panel?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    const focusTarget = els.vapiPublicKey?.value.trim() ? els.callButton : null;
    focusTarget?.focus({ preventScroll: true });
  }, 80);
}

function closeAgentWorkspace() {
  document.body.classList.remove("agent-open");
  els.agentDock?.setAttribute("aria-hidden", "true");
}

async function initializeVad() {
  if (state.vad) return;
  setVad("ready", "Loading voice listener");
  const silero = endpointingPreset(state.endpointingMode).silero;
  state.vad = await MicVAD.new({
    model: "v5",
    baseAssetPath: "/vad/",
    onnxWASMBasePath: "/vad/",
    startOnLoad: false,
    processorType: "auto",
    positiveSpeechThreshold: silero.positiveSpeechThreshold,
    negativeSpeechThreshold: silero.negativeSpeechThreshold,
    redemptionMs: silero.redemptionMs,
    preSpeechPadMs: silero.preSpeechPadMs,
    minSpeechMs: silero.minSpeechMs,
    submitUserSpeechOnPause: false,
    ortConfig: (ort) => {
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;
      ort.env.logLevel = "error";
    },
    onFrameProcessed: ({ isSpeech }) => {
      els.meterBar.style.width = `${Math.max(2, Math.min(100, Math.round(isSpeech * 100)))}%`;
    },
    onSpeechStart: () => {
      if (!state.callActive) return;
      markCallActivity("user_vad_start");
      clearIdleTimers();
      if (state.assistantSpeaking) {
        beginBargeIn("silero_vad");
        recordTraceEvent("barge_in_vad_start", { turn: state.metrics.turn || state.completedTurn + 1 });
        state.vapi?.setMuted(false);
        setConnection("recording", "Interrupting assistant");
        setVad("speech", "Barge-in detected");
        setStage("vad", "Barge-in");
        return;
      }
      state.metrics = { ...freshMetrics(), turn: state.metrics.turn + 1 };
      state.metrics.speechStartedAt = performance.now();
      recordTraceEvent("vad_speech_start", { turn: state.metrics.turn });
      state.vapi?.setMuted(false);
      setConnection("recording", "Customer speaking");
      setVad("speech", "Speech detected");
      setStage("vad", "Listening");
    },
    onSpeechRealStart: () => {
      recordTraceEvent("vad_speech_confirmed", { turn: state.metrics.turn });
      setVad("speech", "Speech confirmed");
    },
    onSpeechEnd: () => {
      if (!state.callActive) return;
      markCallActivity("user_vad_end", { resetPrompt: false });
      state.metrics.speechEndedAt = performance.now();
      state.metrics.vadMs = Math.round(state.metrics.speechEndedAt - state.metrics.speechStartedAt);
      recordTraceEvent("vad_speech_end", { turn: state.metrics.turn, vadMs: state.metrics.vadMs });
      state.vapi?.setMuted(false);
      setConnection("online", "Processing");
      setVad("ready", "Speech ended");
      setStage("stt", "Understanding");
      renderMetrics();
    },
    onVADMisfire: () => {
      recordTraceEvent("vad_misfire", { turn: state.metrics.turn });
      state.vapi?.setMuted(false);
      setVad("ready", "Listening");
    },
  });
}

function handleVapiMessage(message) {
  if (message.type === "transcript") handleTranscript(message);
  if (message.type === "tool-calls" || message.type === "function-call") handleToolCall(message);
  if (message.type === "tool-calls-result" || message.type === "function-call-result") handleToolResult(message);
  if (message.type === "speech-update") handleSpeechUpdate(message);
}

function handleSpeechUpdate(message) {
  const role = message.role || message.speaker || message.speechUpdate?.role || "";
  const status = message.status || message.state || message.speechUpdate?.status || "";
  recordTraceEvent("speech_update", { role, status });
  const normalizedRole = String(role).toLowerCase();
  const normalizedStatus = String(status).toLowerCase();
  if (normalizedRole === "user" && (normalizedStatus.includes("start") || normalizedStatus.includes("speech"))) {
    beginBargeIn("vapi_speech_update");
  }
  if (normalizedRole === "assistant" && (normalizedStatus.includes("stop") || normalizedStatus.includes("end"))) {
    markBargeInStopped("vapi_speech_update");
  }
}

function handleTranscript(message) {
  const text = message.transcript || "";
  if (!text) return;

  if (message.role === "user") {
    markCallActivity("user_transcript");
    clearIdleTimers();
    const interruptionTranscript = state.assistantSpeaking || state.bargeIn.active;
    const firstInterruptionTranscript = interruptionTranscript && !state.bargeIn.capturedAt;
    if (interruptionTranscript) {
      beginBargeIn("user_transcript", text);
      markBargeInCaptured(text, "user_transcript");
    }
    if (firstInterruptionTranscript) {
      startReplacementTurn();
    } else {
      ensureTurnStarted();
    }
    state.userText = text;
    recordTraceEvent("transcript", {
      role: "user",
      transcriptType: message.transcriptType || "partial",
      text,
      turn: state.metrics.turn,
    });
    els.transcriptText.textContent = text;
    if (els.simpleTranscript) els.simpleTranscript.textContent = text;
    if (message.transcriptType === "final") {
      state.metrics.userFinalAt = performance.now();
      if (!state.metrics.speechEndedAt) {
        state.metrics.speechEndedAt = state.metrics.userFinalAt;
        state.metrics.vadMs = state.metrics.speechStartedAt
          ? Math.round(state.metrics.speechEndedAt - state.metrics.speechStartedAt)
          : 0;
      }
      if (state.metrics.speechEndedAt) {
        state.metrics.sttMs = Math.round(state.metrics.userFinalAt - state.metrics.speechEndedAt);
      }
      if (state.bargeIn.active) {
        markBargeInCaptured(text, "user_transcript_final");
        if (state.bargeIn.stoppedAt) state.bargeIn.active = false;
      }
      setStage("llm", "Answering");
      renderMetrics();
      refreshLedger();
    }
  }

  if (message.role === "assistant") {
    state.answer = text;
    recordTraceEvent("transcript", {
      role: "assistant",
      transcriptType: message.transcriptType || "partial",
      text,
      turn: state.metrics.turn,
    });
    els.answerText.textContent = text;
    if (els.simpleAnswer) els.simpleAnswer.textContent = text;
    if (!state.metrics.assistantFirstAt) state.metrics.assistantFirstAt = performance.now();
    updateLatencyDerived();
    renderMetrics();
    if (state.metrics.speechEndedAt && !state.metrics.totalMs) {
      state.metrics.totalMs = Math.round(state.metrics.assistantFirstAt - state.metrics.speechEndedAt);
      renderMetrics();
    }
    if (message.transcriptType === "final" && !state.bargeIn.active) completeLedgerTurn();
  }
}

function ensureTurnStarted() {
  if (state.metrics.turn > state.completedTurn && state.metrics.speechStartedAt) return;
  state.metrics = { ...freshMetrics(), turn: Math.max(state.metrics.turn, state.completedTurn) + 1 };
  state.metrics.speechStartedAt = performance.now();
}

function startReplacementTurn() {
  if (state.metrics.turn > state.completedTurn) {
    recordTraceEvent("turn_interrupted", {
      interruptedTurn: state.metrics.turn,
      replacementTurn: state.metrics.turn + 1,
      partialAnswer: state.answer,
    });
    state.completedTurn = state.metrics.turn;
  }
  state.metrics = {
    ...freshMetrics(),
    turn: state.completedTurn + 1,
    speechStartedAt: state.bargeIn.startedAt || performance.now(),
  };
}

function completeLedgerTurn() {
  if (!state.userText) return;
  if (!state.metrics.turn) {
    state.metrics = { ...state.metrics, turn: state.completedTurn + 1 };
  }
  if (state.metrics.turn <= state.completedTurn) return;

  updateLatencyDerived();
  state.completedTurn = state.metrics.turn;
  state.turnCount += 1;
  els.turnCount.textContent = `${state.turnCount} ${state.turnCount === 1 ? "turn" : "turns"}`;
  recordTraceEvent("turn_complete", {
    turn: state.metrics.turn,
    userText: state.userText,
    answer: state.answer,
    metrics: state.metrics,
    bargeIn: state.bargeIn,
  });
  renderMetrics();
  refreshLedger();
}

async function handleToolCall(message) {
  const calls = message.toolCallList || message.toolCalls || [];
  const call = calls[0] || message.functionCall;
  if (!call) return;
  const fn = call.function || call;
  const args = parseJson(fn.arguments) || fn.parameters || {};
  const toolName = fn.name || call.name || "lookup_inventory";
  if (!state.metrics.toolStartedAt) {
    state.metrics.toolStartedAt = performance.now();
    renderMetrics();
  }
  recordTraceEvent("tool_call", {
    id: call.id || call.toolCallId,
    name: toolName,
    query: args.query || args.item || "",
    arguments: args,
  });
  els.toolItem.textContent = args.query ? `Looking up ${args.query}` : "Looking up";
  els.toolAisle.textContent = "...";
  els.toolStock.textContent = "...";
  els.toolMatch.textContent = "...";
  setStage("llm", toolName === "search_knowledge" ? "Knowledge search" : "Inventory tool call");

  if (args.query) {
    try {
      const path = toolName === "search_knowledge" ? "/api/knowledge/answer" : "/api/inventory/lookup";
      const response = await fetch(previewBackendUrl(path), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: args.query }),
      });
      if (response.ok) renderTool(await response.json());
    } catch {
      // Vapi still owns the authoritative server-side tool call.
    }
  }
}

function handleToolResult(message) {
  const results = message.results || message.toolCallResults || [message];
  for (const result of results) {
    const payload = parseJson(result.result) || parseJson(result.content) || result.result;
    if (payload && typeof payload === "object") {
      state.metrics.toolEndedAt = performance.now();
      state.metrics.toolMs = durationMs(state.metrics.toolStartedAt, state.metrics.toolEndedAt);
      recordTraceEvent("tool_result", payload);
      recordEvidenceGate(payload);
      renderTool(payload);
      renderMetrics();
    }
  }
}

function recordEvidenceGate(payload) {
  const gate = payload?.answerGate;
  if (!gate) return;
  recordTraceEvent("evidence_gate", {
    query: payload.query,
    status: gate.status,
    action: gate.action,
    reason: gate.reason,
    faithfulnessScore: gate.faithfulnessScore,
    faithfulnessVerdict: gate.faithfulnessVerdict,
    answerable: payload.answerable,
    signatures: gate.evidenceSignatures || [],
  });
}
function parseJson(value) {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function renderTool(result) {
  if (result?.tool === "search_knowledge") {
    renderKnowledgeTool(result);
    return;
  }

  setToolReadoutLabels("Catalog lookup", ["Item", "Aisle", "Stock", "Match"]);

  if (!result?.found) {
    els.toolItem.textContent = "Not found";
    els.toolAisle.textContent = "-";
    els.toolStock.textContent = "-";
    els.toolMatch.textContent = `${Math.round((result?.score || 0) * 100)}%`;
    renderCatalogInsight(result);
    return;
  }

  renderCatalogInsight(result);

  if (result.matchType === "recommendation" && renderBestOption(result)) return;

  if (Array.isArray(result.matches) && result.matches.length > 1) {
    const preview = result.matches
      .slice(0, 3)
      .map((item) => item.name)
      .join("; ");
    const aisles = Array.isArray(result.aisles) ? result.aisles.join(", ") : "-";
    const totalStock = result.matches.reduce((sum, item) => sum + Number(item.stock || 0), 0);
    els.toolItem.textContent = preview;
    els.toolAisle.textContent = aisles;
    els.toolStock.textContent = `${totalStock} total units`;
    els.toolMatch.textContent = `${Math.round((result.score || 0) * 100)}%`;
    return;
  }

  const item = result.item;
  els.toolItem.textContent = item.name;
  els.toolAisle.textContent = `${item.aisle} / ${item.bay}`;
  els.toolStock.textContent = item.stock === 1 ? "1 unit" : `${item.stock} units`;
  els.toolMatch.textContent = `${Math.round((result.score || 0) * 100)}%`;
}

function renderBestOption(result) {
  const best = result?.bestOption?.item;
  if (!best) return false;
  setToolReadoutLabels("Catalog lookup", ["Recommended", "Aisle", "Reviews", "Score"]);
  const reviews = best.customerReviewSummary || {};
  els.toolItem.textContent = best.name;
  els.toolAisle.textContent = `${best.aisle} / ${best.bay}`;
  els.toolStock.textContent = `${reviews.rating || "-"} stars / ${reviews.reviewCount || 0} reviews`;
  els.toolMatch.textContent = `${Math.round(Number(result.bestOption.score || 0) * 100)}%`;
  return true;
}
function setToolReadoutLabels(title, labels) {
  const displayTitle = {
    lookup_inventory: "Catalog lookup",
    search_knowledge: "Policy lookup",
  }[title] || title;
  els.toolTitle.textContent = displayTitle;
  [els.toolItemLabel, els.toolAisleLabel, els.toolStockLabel, els.toolMatchLabel].forEach((label, index) => {
    label.textContent = labels[index];
  });
}

function syncRagLabFromKnowledgeTool(result) {
  if (state.ragLabManualRunning || !result?.query) return;
  state.latestRagLab = result;
  if (els.ragQueryInput) els.ragQueryInput.value = result.query;
  renderRagLab(result);
  const draftAnswer = draftAnswerFromRag(result);
  if (els.ragAnswerInput && draftAnswer && (!els.ragAnswerInput.value.trim() || els.ragAnswerInput.value === state.latestRagDraftAnswer)) {
    els.ragAnswerInput.value = draftAnswer;
    state.latestRagDraftAnswer = draftAnswer;
  }
  if (result.faithfulness) renderFaithfulness(result.faithfulness);
  runRagAblation(result.query);
  if (!result.faithfulness) runFaithfulnessGrade(result.query);
}

function renderKnowledgeTool(result) {
  const gate = result?.answerGate;
  if (gate) {
    setToolReadoutLabels("Policy lookup", ["Gate", "Action", "Sources", "Faithfulness"]);
    const sources = Array.isArray(result.sources) ? result.sources : [];
    const score = gate.faithfulnessScore ?? result.faithfulness?.faithfulnessScore;
    els.toolItem.textContent = gate.status || "review";
    els.toolAisle.textContent = gate.action || "-";
    els.toolStock.textContent = sources.length ? sources.join(", ") : "0 sources";
    els.toolMatch.textContent = `${gate.faithfulnessVerdict || result.faithfulness?.verdict || "not run"} / ${formatPercent(score)}`;
    if (result.speechAnswer && !state.assistantSpeaking) els.answerText.textContent = result.speechAnswer;
    syncRagLabFromKnowledgeTool(result);
    return;
  }

  setToolReadoutLabels("Policy lookup", ["Evidence", "Method", "Sources", "Validation"]);

  if (!result?.found || !Array.isArray(result.results) || result.results.length === 0) {
    els.toolItem.textContent = "Knowledge not found";
    els.toolAisle.textContent = result?.retrieval?.method || "-";
    els.toolStock.textContent = "-";
    els.toolMatch.textContent = "-";
    syncRagLabFromKnowledgeTool(result);
    return;
  }

  const sources = result.results.map((entry) => entry.source).join(", ");
  const titles = [...new Set(result.results.map((entry) => `${entry.title}: ${entry.section}`))].slice(0, 2).join("; ");
  const confidence = Number(result.retrieval?.confidence ?? result.results[0]?.score ?? 0);
  const validation = result.validation?.grounded ? "grounded" : result.validation?.status || "review";
  els.toolItem.textContent = titles;
  els.toolAisle.textContent = result.retrieval?.method || "KB";
  els.toolStock.textContent = sources;
  els.toolMatch.textContent = `${validation} / ${Math.round(confidence * 100)}%`;
  syncRagLabFromKnowledgeTool(result);
}
function formatPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number * 100)}%` : "-";
}

function renderRagSampleButtons() {
  if (!els.ragSampleRows) return;
  els.ragSampleRows.replaceChildren(
    ...sampleRagQuestions.map((question) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = question;
      button.addEventListener("click", () => {
        if (els.ragQueryInput) els.ragQueryInput.value = question;
        runRagLab(question);
      });
      return button;
    }),
  );
}

async function runRagLab(query = els.ragQueryInput?.value) {
  const cleanQuery = String(query || "").trim();
  if (!cleanQuery || !els.ragRunButton) return;

  els.ragRunButton.disabled = true;
  els.ragRunButton.textContent = "Running";
  if (els.ragPlanRows) els.ragPlanRows.textContent = "Planning query.";
  if (els.ragEvidenceRows) els.ragEvidenceRows.textContent = "Retrieving evidence.";
  if (els.ragContractRows) els.ragContractRows.textContent = "Building contract.";

  try {
    const startedAt = performance.now();
    const response = await fetch(previewBackendUrl("/api/knowledge/answer"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: cleanQuery, limit: 4 }),
    });
    const latencyMs = Math.round(performance.now() - startedAt);
    if (!response.ok) throw new Error("RAG search failed");
    const payload = await response.json();
    payload.retrievalLatencyMs = latencyMs;
    state.latestRagLab = payload;
    renderRagLab(payload);
    state.ragLabManualRunning = true;
    renderTool(payload);
    state.ragLabManualRunning = false;
    const draftAnswer = draftAnswerFromRag(payload);
    if (els.ragAnswerInput && draftAnswer && (!els.ragAnswerInput.value.trim() || els.ragAnswerInput.value === state.latestRagDraftAnswer)) {
      els.ragAnswerInput.value = draftAnswer;
      state.latestRagDraftAnswer = draftAnswer;
    }
    recordEvidenceGate(payload);
    runRagAblation(cleanQuery);
    if (payload.faithfulness) renderFaithfulness(payload.faithfulness);
    else runFaithfulnessGrade(cleanQuery);
    recordTraceEvent("rag_lab", {
      query: cleanQuery,
      found: Boolean(payload.found),
      confidence: payload.retrieval?.confidence,
      margin: payload.retrieval?.margin,
      validation: payload.validation?.status,
      latencyMs,
      sources: payload.sources || [],
    });
  } catch (error) {
    renderRagLabError(error);
  } finally {
    els.ragRunButton.disabled = false;
    els.ragRunButton.textContent = "Run";
  }
}

async function runRagAblation(query) {
  if (!els.ragAblationRows) return;
  els.ragAblationRows.textContent = "Comparing retrieval variants.";
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/rag-ablation"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, limit: 4 }),
    });
    if (!response.ok) throw new Error("RAG ablation failed");
    const payload = await response.json();
    renderRagAblation(payload);
    recordTraceEvent("rag_ablation", {
      query,
      winner: payload.winner,
      variants: (payload.variants || []).map((variant) => ({
        id: variant.id,
        found: variant.found,
        confidence: variant.confidence,
        margin: variant.margin,
        latencyMs: variant.latencyMs,
      })),
    });
  } catch (error) {
    const empty = document.createElement("div");
    empty.className = "trace-empty";
    empty.textContent = readableError(error);
    els.ragAblationRows.replaceChildren(empty);
  }
}

function renderRagAblation(payload) {
  if (!els.ragAblationRows) return;
  const variants = Array.isArray(payload?.variants) ? payload.variants : [];
  if (!variants.length) {
    const empty = document.createElement("div");
    empty.className = "trace-empty";
    empty.textContent = "No retrieval variants returned.";
    els.ragAblationRows.replaceChildren(empty);
    return;
  }

  const maxConfidence = Math.max(...variants.map((variant) => Number(variant.confidence || 0)), 0.01);
  els.ragAblationRows.replaceChildren(
    ...variants.map((variant) => {
      const row = document.createElement("article");
      row.className = `rag-ablation-row ${variant.id === payload.winner ? "winner" : variant.found ? "pass" : "review"}`;
      const title = document.createElement("strong");
      title.textContent = `${variant.label}${variant.id === payload.winner ? " - winner" : ""}`;
      const meta = document.createElement("span");
      meta.textContent = `${variant.supportStatus || "review"} / confidence ${formatPercent(variant.confidence)} / margin ${formatPercent(variant.margin)} / ${variant.sourceCount || 0} sources / ${formatMs(variant.latencyMs)}`;
      const sources = document.createElement("span");
      sources.textContent = variant.sources?.length ? variant.sources.join(", ") : "No supported sources";
      const bar = document.createElement("span");
      bar.className = "rag-ablation-bar";
      const fill = document.createElement("span");
      fill.style.width = `${Math.max(4, Math.min(100, (Number(variant.confidence || 0) / maxConfidence) * 100))}%`;
      bar.append(fill);
      const description = document.createElement("p");
      description.textContent = variant.description || "";
      row.append(title, meta, sources, bar, description);
      return row;
    }),
  );
}
function draftAnswerFromRag(payload) {
  const generated = payload?.answerGeneration?.finalAnswer || payload?.speechAnswer || payload?.answerGeneration?.draftAnswer || "";
  if (generated) return generated.length > 420 ? `${generated.slice(0, 417).trim()}...` : generated;
  const first = Array.isArray(payload?.evidence) ? payload.evidence[0] : Array.isArray(payload?.results) ? payload.results[0] : null;
  const text = first?.compressedText || first?.text || "";
  if (!text) return "";
  return text.length > 260 ? `${text.slice(0, 257).trim()}...` : text;
}
async function runFaithfulnessGrade(query = els.ragQueryInput?.value) {
  if (!els.ragFaithfulnessRows || !els.ragAnswerInput) return;
  const cleanQuery = String(query || "").trim();
  const answer = els.ragAnswerInput.value.trim();
  if (!cleanQuery || !answer) return;
  els.ragFaithfulnessRows.textContent = "Grading answer against evidence signatures.";
  if (els.ragFaithfulnessButton) els.ragFaithfulnessButton.disabled = true;
  try {
    const evidence = Array.isArray(state.latestRagLab?.evidence) ? state.latestRagLab.evidence : undefined;
    const response = await fetch(previewBackendUrl("/api/evaluation/faithfulness"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: cleanQuery, answer, evidence, limit: 4 }),
    });
    if (!response.ok) throw new Error("Faithfulness grading failed");
    const payload = await response.json();
    renderFaithfulness(payload);
    recordTraceEvent("faithfulness_eval", {
      query: cleanQuery,
      verdict: payload.verdict,
      score: payload.faithfulnessScore,
      grounded: payload.grounded,
      issues: payload.issues || [],
      signatures: payload.usedEvidenceSignatures || [],
    });
  } catch (error) {
    renderFaithfulnessError(error);
  } finally {
    if (els.ragFaithfulnessButton) els.ragFaithfulnessButton.disabled = false;
  }
}

function renderFaithfulness(payload) {
  const claims = Array.isArray(payload?.sentenceClaims) ? payload.sentenceClaims : [];
  const signatures = Array.isArray(payload?.usedEvidenceSignatures) ? payload.usedEvidenceSignatures : [];
  if (els.ragFaithfulnessVerdict) els.ragFaithfulnessVerdict.textContent = payload?.verdict || "-";
  if (els.ragFaithfulnessScore) els.ragFaithfulnessScore.textContent = formatPercent(payload?.faithfulnessScore);
  if (els.ragFaithfulnessClaims) els.ragFaithfulnessClaims.textContent = String(claims.length || 0);
  if (els.ragFaithfulnessSignatures) els.ragFaithfulnessSignatures.textContent = String(signatures.length || 0);
  if (!els.ragFaithfulnessRows) return;
  if (!claims.length) {
    const empty = document.createElement("div");
    empty.className = "trace-empty";
    empty.textContent = (payload?.issues || ["No claims graded."]).join(" / ");
    els.ragFaithfulnessRows.replaceChildren(empty);
    return;
  }
  els.ragFaithfulnessRows.replaceChildren(
    ...claims.map((claim) => {
      const row = document.createElement("article");
      row.className = `faithfulness-row ${claim.status === "supported" ? "pass" : claim.status === "unsupported" ? "fail" : "review"}`;
      const title = document.createElement("strong");
      title.textContent = `${claim.status} / ${formatPercent(claim.score)}`;
      const sentence = document.createElement("p");
      sentence.textContent = claim.sentence || "";
      const meta = document.createElement("span");
      meta.textContent = `${claim.source || "no source"} / ${claim.evidenceSignature || "no signature"}`;
      const terms = document.createElement("span");
      const overlap = Array.isArray(claim.overlapTerms) && claim.overlapTerms.length ? claim.overlapTerms.join(", ") : "no overlap terms";
      terms.textContent = `terms: ${overlap}`;
      row.append(title, sentence, meta, terms);
      return row;
    }),
  );
}

function renderFaithfulnessError(error) {
  if (els.ragFaithfulnessVerdict) els.ragFaithfulnessVerdict.textContent = "failed";
  if (els.ragFaithfulnessScore) els.ragFaithfulnessScore.textContent = "-";
  if (els.ragFaithfulnessClaims) els.ragFaithfulnessClaims.textContent = "0";
  if (els.ragFaithfulnessSignatures) els.ragFaithfulnessSignatures.textContent = "0";
  if (!els.ragFaithfulnessRows) return;
  const empty = document.createElement("div");
  empty.className = "trace-empty";
  empty.textContent = readableError(error);
  els.ragFaithfulnessRows.replaceChildren(empty);
}
function renderRagLab(payload) {
  const analysis = payload?.queryAnalysis || {};
  const retrieval = payload?.retrieval || {};
  const validation = payload?.validation || {};
  const sources = Array.isArray(payload?.sources) ? payload.sources : [];

  if (els.ragIntentMetric) els.ragIntentMetric.textContent = analysis.intent || "-";
  if (els.ragConfidenceMetric) els.ragConfidenceMetric.textContent = formatPercent(retrieval.confidence);
  if (els.ragMarginMetric) els.ragMarginMetric.textContent = formatPercent(retrieval.margin);
  if (els.ragValidationMetric) els.ragValidationMetric.textContent = validation.grounded ? "grounded" : validation.status || "review";
  if (els.ragLatencyMetric) els.ragLatencyMetric.textContent = formatMs(payload.retrievalLatencyMs);
  if (els.ragSourcesMetric) els.ragSourcesMetric.textContent = String(sources.length || 0);

  renderRagPlan(payload);
  renderRagEvidence(payload);
  renderRagContract(payload);
  renderEvidenceGate(payload);
}

function renderRagLabError(error) {
  const message = readableError(error);
  if (els.ragIntentMetric) els.ragIntentMetric.textContent = "-";
  if (els.ragConfidenceMetric) els.ragConfidenceMetric.textContent = "-";
  if (els.ragMarginMetric) els.ragMarginMetric.textContent = "-";
  if (els.ragValidationMetric) els.ragValidationMetric.textContent = "failed";
  if (els.ragLatencyMetric) els.ragLatencyMetric.textContent = "-";
  if (els.ragSourcesMetric) els.ragSourcesMetric.textContent = "0";
  renderEvidenceGateError(message);
  renderKeyValueRows(els.ragPlanRows, [{ label: "Error", value: message }]);
  if (els.ragEvidenceRows) {
    const empty = document.createElement("div");
    empty.className = "trace-empty";
    empty.textContent = message;
    els.ragEvidenceRows.replaceChildren(empty);
  }
  if (els.ragContractRows) {
    const empty = document.createElement("div");
    empty.className = "trace-empty";
    empty.textContent = "No answer contract.";
    els.ragContractRows.replaceChildren(empty);
  }
}


function renderEvidenceGate(payload) {
  const gate = payload?.answerGate || {};
  const faithfulness = payload?.faithfulness || {};
  const status = gate.status || "not run";
  const action = gate.action || "-";
  const score = gate.faithfulnessScore ?? faithfulness.faithfulnessScore;
  const latency = gate.latencyMs;

  if (els.ragGateStatus) els.ragGateStatus.textContent = status;
  if (els.ragGateAction) els.ragGateAction.textContent = action;
  if (els.ragGateScore) els.ragGateScore.textContent = formatPercent(score);
  if (els.ragGateLatency) els.ragGateLatency.textContent = formatMs(latency);
  if (!els.ragGateRows) return;

  if (!payload?.answerGate) {
    const empty = document.createElement("div");
    empty.className = "trace-empty";
    empty.textContent = "Retrieval only. Run the gated answer endpoint to make a speak/block decision.";
    els.ragGateRows.replaceChildren(empty);
    return;
  }

  const row = document.createElement("article");
  row.className = `gate-row ${status === "approved" ? "approved" : status === "blocked" ? "blocked" : "review"}`;
  const title = document.createElement("strong");
  title.textContent = `${status} / ${action}`;
  const answer = document.createElement("p");
  answer.textContent = payload.answerGeneration?.finalAnswer || payload.speechAnswer || "No final answer.";
  const meta = document.createElement("span");
  meta.textContent = `${gate.reason || "no reason"} / verdict ${gate.faithfulnessVerdict || faithfulness.verdict || "not run"} / score ${formatPercent(score)}`;
  const signatures = document.createElement("span");
  const signatureList = Array.isArray(gate.evidenceSignatures) ? gate.evidenceSignatures : [];
  signatures.textContent = signatureList.length ? `signatures: ${signatureList.join(", ")}` : "signatures: none";
  row.append(title, answer, meta, signatures);
  els.ragGateRows.replaceChildren(row);
}

function renderEvidenceGateError(message) {
  if (els.ragGateStatus) els.ragGateStatus.textContent = "failed";
  if (els.ragGateAction) els.ragGateAction.textContent = "-";
  if (els.ragGateScore) els.ragGateScore.textContent = "-";
  if (els.ragGateLatency) els.ragGateLatency.textContent = "-";
  if (!els.ragGateRows) return;
  const empty = document.createElement("div");
  empty.className = "trace-empty";
  empty.textContent = message;
  els.ragGateRows.replaceChildren(empty);
}
function renderRagPlan(payload) {
  const analysis = payload?.queryAnalysis || {};
  const retrieval = payload?.retrieval || {};
  const validation = payload?.validation || {};
  renderKeyValueRows(els.ragPlanRows, [
    { label: "Intent", value: analysis.intent || "-" },
    { label: "Topics", value: analysis.topics || [] },
    { label: "Transforms", value: analysis.transformedQueries || [] },
    { label: "Sub-questions", value: analysis.subQuestions || [] },
    { label: "Filters", value: analysis.filters || {} },
    { label: "Pipeline", value: retrieval.pipeline || [] },
    { label: "Issues", value: validation.issues?.length ? validation.issues : ["none"] },
  ]);
}

function renderRagEvidence(payload) {
  if (!els.ragEvidenceRows) return;
  const results = Array.isArray(payload?.results) ? payload.results : [];
  if (!results.length) {
    const empty = document.createElement("div");
    empty.className = "trace-empty";
    empty.textContent = payload?.guidance || "No supported evidence returned.";
    els.ragEvidenceRows.replaceChildren(empty);
    return;
  }

  els.ragEvidenceRows.replaceChildren(
    ...results.map((entry) => {
      const card = document.createElement("article");
      card.className = "rag-card";
      const title = document.createElement("strong");
      title.textContent = `${entry.source} / ${entry.title || "Knowledge"}`;
      const meta = document.createElement("span");
      const metadata = entry.metadata || {};
      meta.textContent = `${entry.section || "section"} / ${metadata.document_type || "doc"} / ${metadata.top_topic || "topic"} / ${metadata.updated_at || "freshness unknown"}`;
      const score = document.createElement("span");
      score.textContent = `score ${formatPercent(entry.score)} / coverage ${formatPercent(entry.coverage)} / ${entry.evidenceSignature || "no signature"}`;
      const text = document.createElement("p");
      text.textContent = entry.compressedText || entry.text || "";
      card.append(title, meta, score, text);
      return card;
    }),
  );
}

function renderRagContract(payload) {
  const contract = payload?.promptContract || {};
  renderKeyValueRows(els.ragContractRows, [
    { label: "Role", value: contract.role || "-" },
    { label: "Answer style", value: contract.answerStyle || "-" },
    { label: "Must use evidence", value: contract.mustUseEvidence ? "yes" : "no" },
    { label: "Citations", value: contract.displayCitations || [] },
    { label: "Do not invent", value: contract.mustNotInvent || [] },
    { label: "Spoken citations", value: contract.spokenCitationPolicy || "-" },
  ]);
}

function renderKeyValueRows(container, rows) {
  if (!container) return;
  container.replaceChildren(
    ...rows.map((row) => {
      const item = document.createElement("div");
      item.className = "rag-kv-row";
      const label = document.createElement("span");
      label.textContent = row.label;
      const value = document.createElement("strong");
      value.textContent = formatRagValue(row.value);
      item.append(label, value);
      return item;
    }),
  );
}

function formatRagValue(value) {
  if (Array.isArray(value)) return value.length ? value.join(" / ") : "-";
  if (value && typeof value === "object") return JSON.stringify(value);
  if (value === false) return "false";
  return value || "-";
}
function renderCatalogSummary(summary) {
  const departments = summary?.departments || {};
  const availability = summary?.availability || {};
  els.catalogProductCount.textContent = summary?.totalProducts ?? "-";
  els.catalogDepartmentCount.textContent = Object.keys(departments).length || "-";
  els.catalogLowStock.textContent = availability.lowStock ?? "-";
  els.catalogOutStock.textContent = availability.outOfStock ?? "-";
}

function renderCatalogInsight(result) {
  if (!result?.found) {
    els.catalogEvidence.textContent = result?.reason ? `No supported catalog evidence. Best reason: ${result.reason}.` : "No supported catalog evidence.";
    els.relationRows.textContent = "No alternatives available until a catalog item is matched.";
    return;
  }

  const summary = result.availabilitySummary || {};
  const evidence = Array.isArray(result.retrievalEvidence) ? result.retrievalEvidence : [];
  const topEvidence = evidence
    .slice(0, 3)
    .map((entry) => `#${entry.rank} ${entry.sku} ${Math.round(Number(entry.score || 0) * 100)}% ${entry.reason}`)
    .join(" / ");
  els.catalogEvidence.textContent = topEvidence || `${summary.items || result.itemCount || 1} catalog row matched.`;

  const best = result.bestOption?.item;
  const relations = [
    ...(best ? [{ ...best, relationType: "best overall", relationReason: result.bestOption.reason || "highest recommendation score" }] : []),
    ...(Array.isArray(result.alternatives) ? result.alternatives.slice(0, 3) : []),
    ...(Array.isArray(result.complements) ? result.complements.slice(0, 3) : []),
  ];
  if (!relations.length) {
    els.relationRows.textContent = "No grounded related products for this lookup.";
    return;
  }

  els.relationRows.replaceChildren(
    ...relations.map((item) => {
      const row = document.createElement("div");
      row.className = "relation-row";
      const type = document.createElement("span");
      type.className = "relation-type";
      type.textContent = item.relationType || "related";
      const name = document.createElement("strong");
      name.textContent = item.name;
      const location = document.createElement("span");
      location.textContent = `${item.aisle} / ${item.bay} / ${item.stock} ${Number(item.stock) === 1 ? "unit" : "units"}`;
      const reason = document.createElement("span");
      reason.textContent = item.relationReason || item.subcategory || item.department || "";
      row.append(type, name, location, reason);
      return row;
    }),
  );
}

function renderMetrics() {
  updateLatencyDerived();
  els.metricTotal.textContent = formatMs(state.metrics.totalMs);
  els.metricVad.textContent = formatMs(state.metrics.vadMs);
  els.metricStt.textContent = formatMs(state.metrics.sttMs);
  els.metricTts.textContent = formatMs(state.metrics.voiceMs);
  renderSpeechBenchmark();
}

function renderSpeechBenchmark() {
  updateLatencyDerived();
  if (els.endpointingModeMetric) els.endpointingModeMetric.textContent = endpointingPreset(state.endpointingMode).label;
  els.bargeInCount.textContent = `${state.bargeIn.successes} / ${state.bargeIn.attempts}`;
  els.bargeStopMs.textContent = formatMs(state.bargeIn.stopLatencyMs);
  els.bargeCaptureMs.textContent = formatMs(state.bargeIn.captureLatencyMs);
  els.overlapSavedMs.textContent = formatMs(state.metrics.overlapSavedMs);
  els.bargeStatus.textContent = state.bargeIn.status;

  const rows = [
    { id: "vad", label: "Capture", value: state.metrics.vadMs },
    { id: "stt", label: "Speech", value: state.metrics.sttMs },
    { id: "tool", label: "Lookup", value: state.metrics.toolMs },
    { id: "llm", label: "Answer", value: state.metrics.llmMs },
    { id: "tts", label: "Voice", value: state.metrics.ttsFirstMs },
  ].filter((row) => Number.isFinite(row.value));

  if (!rows.length) {
    els.waterfallRows.textContent = "Waiting for the next turn.";
    return;
  }

  const max = Math.max(...rows.map((row) => row.value), 1);
  els.waterfallRows.replaceChildren(
    ...rows.map((row) => {
      const item = document.createElement("div");
      item.className = "waterfall-row";
      const label = document.createElement("span");
      label.textContent = row.label;
      const track = document.createElement("span");
      track.className = "bar-track";
      const fill = document.createElement("span");
      fill.className = `bar-fill ${row.id}`;
      fill.style.width = `${Math.max(4, Math.min(100, (Number(row.value || 0) / max) * 100))}%`;
      track.append(fill);
      const value = document.createElement("span");
      value.textContent = formatMs(row.value);
      item.append(label, track, value);
      return item;
    }),
  );
}

async function refreshLedger() {
  const callMs = state.callStartedAt ? Math.round(performance.now() - state.callStartedAt) : 0;
  try {
    const response = await fetch(previewBackendUrl("/api/cost-estimate"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        callMs,
        userText: state.userText,
        answer: state.answer,
        totalLatencyMs: state.metrics.totalMs,
      }),
    });
    if (!response.ok) throw new Error("Cost endpoint unavailable");
    const costs = await response.json();
    renderLedger(costs);
    recordTraceEvent("ledger", costs);
  } catch {
    els.assumptionAudio.textContent = "Ledger endpoint unavailable";
  }
}

function renderLedger(costs) {
  if (!costs?.rows) return;
  state.latestLedger = costs;
  const maxLatency = Math.max(...costs.rows.map((row) => row.latencyMs || 0), 1);
  const maxCost = Math.max(...costs.rows.map((row) => row.per1000 || 0), 0.0001);
  els.ledgerRows.replaceChildren(
    ...costs.rows.map((row) => {
      const item = document.createElement("article");
      item.className = "ledger-row";

      const top = document.createElement("div");
      top.className = "ledger-top";
      const nameWrap = document.createElement("div");
      const name = document.createElement("p");
      name.className = "ledger-name";
      name.textContent = row.label;
      const mode = document.createElement("p");
      mode.className = "ledger-mode";
      mode.textContent = row.mode;
      nameWrap.append(name, mode);

      const cost = document.createElement("div");
      cost.className = "ledger-cost";
      const costStrong = document.createElement("strong");
      costStrong.textContent = formatMoney(row.cost);
      const per = document.createElement("span");
      per.textContent = `${formatMoney(row.per1000)} / 1k`;
      cost.append(costStrong, per);
      top.append(nameWrap, cost);

      const bars = document.createElement("div");
      bars.className = "bar-pair";
      bars.append(
        barLine("Latency", row.latencyMs, maxLatency, false),
        barLine("Cost", row.per1000, maxCost, true),
      );
      item.append(top, bars);
      return item;
    }),
  );

  const assumptions = costs.assumptions || {};
  els.assumptionAudio.textContent = `Call ${assumptions.callSeconds || "-"}s / speech ${assumptions.audioOutputSeconds || "-"}s`;
  els.assumptionTokens.textContent = `Tokens ${assumptions.promptTokens || "-"} in / ${assumptions.completionTokens || "-"} out`;
  updateTracePanel();
}

function barLine(label, value, max, isCost) {
  const line = document.createElement("div");
  line.className = "bar-line";
  const left = document.createElement("span");
  left.textContent = label;
  const track = document.createElement("span");
  track.className = "bar-track";
  const fill = document.createElement("span");
  fill.className = `bar-fill${isCost ? " cost" : ""}`;
  fill.style.width = `${Math.max(4, Math.min(100, (Number(value || 0) / max) * 100))}%`;
  track.append(fill);
  const right = document.createElement("span");
  right.textContent = isCost ? formatMoney(value) : formatMs(value);
  line.append(left, track, right);
  return line;
}

async function finalizeCall() {
  const hadActiveCall = state.callActive || Boolean(state.callStartedAt);
  state.callEndedAt = performance.now();
  state.lastCallSeconds = state.callStartedAt ? Math.max(0, Math.round((state.callEndedAt - state.callStartedAt) / 1000)) : 0;
  clearIdleTimers();
  if (state.trace && state.trace.status === "recording") {
    recordTraceEvent("call_finalize", { turnCount: state.turnCount, endReason: state.callEndReason });
  }
  if (state.vad) {
    const vad = state.vad;
    state.vad = null;
    try {
      await vad.destroy();
    } catch {
      // The microphone may already be closed by Vapi.
    }
  }
  if (state.vapi) {
    state.vapi.removeAllListeners();
    state.vapi = null;
  }
  if (state.ledgerTimer) {
    window.clearInterval(state.ledgerTimer);
    state.ledgerTimer = null;
  }
  await refreshLedger();
  await saveCurrentTrace("completed");
  state.callActive = false;
  state.starting = false;
  state.assistantSpeaking = false;
  els.meterBar.style.width = "0%";
  setCallUi(false);
  setConnection("offline", "Offline");
  setVad("idle", "Voice idle");
  setStage("idle", "Idle");
  stopSimpleTimer();
  if (hadActiveCall) showFeedbackPanel();
}

async function loadBenchmarkCases() {
  if (!els.benchmarkGroupSelect) return;
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/suite/cases"));
    if (!response.ok) throw new Error("Benchmark cases unavailable");
    const payload = await response.json();
    const groups = Array.isArray(payload.groups) ? payload.groups : [];
    const current = els.benchmarkGroupSelect.value;
    els.benchmarkGroupSelect.replaceChildren(
      optionNode("", "All groups"),
      ...groups.map((group) => optionNode(group, group)),
    );
    els.benchmarkGroupSelect.value = groups.includes(current) ? current : "";
  } catch {
    els.benchmarkGroupSelect.replaceChildren(optionNode("", "All groups"));
  }
}

function optionNode(value, label) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  return option;
}

async function loadLatestBenchmark() {
  if (!els.benchmarkCaseRows) return;
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/suite/latest"));
    if (!response.ok) throw new Error("Latest benchmark unavailable");
    const payload = await response.json();
    if (payload?.found === false) return;
    renderBenchmarkSuite(payload);
  } catch {
    // The suite may not have been run yet.
  }
}

async function runBenchmarkSuite() {
  if (!els.benchmarkRunButton) return;
  els.benchmarkRunButton.disabled = true;
  els.benchmarkRunButton.textContent = "Running";
  if (els.benchmarkCaseRows) els.benchmarkCaseRows.textContent = "Running benchmark cases.";
  if (els.benchmarkGroupRows) els.benchmarkGroupRows.textContent = "Scoring groups.";
  try {
    const limit = Number.parseInt(els.benchmarkLimitInput?.value || "", 10);
    const group = els.benchmarkGroupSelect?.value || "";
    const response = await fetch(previewBackendUrl("/api/evaluation/suite/run"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        groups: group ? [group] : [],
        limit: Number.isFinite(limit) && limit > 0 ? limit : null,
        includePayloads: false,
        save: true,
      }),
    });
    if (!response.ok) throw new Error("Benchmark suite failed");
    const payload = await response.json();
    renderBenchmarkSuite(payload);
    recordTraceEvent("benchmark_suite", {
      runId: payload.runId,
      total: payload.summary?.total,
      passed: payload.summary?.passed,
      passRate: payload.summary?.passRate,
      p95: payload.summary?.latency?.voiceP95Ms,
      per1000: payload.summary?.cost?.per1000VapiStack,
    });
  } catch (error) {
    renderBenchmarkError(error);
  } finally {
    els.benchmarkRunButton.disabled = false;
    els.benchmarkRunButton.textContent = "Run suite";
  }
}

function renderBenchmarkSuite(payload) {
  state.latestBenchmark = payload;
  const summary = payload?.summary || {};
  const latency = summary.latency || {};
  const cost = summary.cost || {};
  const gateMatrix = summary.gateConfusion || {};
  const matrixText = Object.entries(gateMatrix)
    .map(([key, value]) => `${key.replaceAll("expected:", "e:").replaceAll("observed:", "o:")} ${value}`)
    .join(" / ");

  if (els.benchmarkCasesMetric) els.benchmarkCasesMetric.textContent = `${summary.passed ?? 0} / ${summary.total ?? 0}`;
  if (els.benchmarkPassMetric) els.benchmarkPassMetric.textContent = formatPercent(summary.passRate);
  if (els.benchmarkRuntimeMetric) els.benchmarkRuntimeMetric.textContent = formatMs(latency.runtimeP95Ms);
  if (els.benchmarkVoiceMetric) els.benchmarkVoiceMetric.textContent = formatMs(latency.voiceP95Ms);
  if (els.benchmarkCostMetric) els.benchmarkCostMetric.textContent = formatMoney(cost.per1000VapiStack || 0);
  if (els.benchmarkGateMetric) els.benchmarkGateMetric.textContent = matrixText || "-";
  if (els.benchmarkRunId) els.benchmarkRunId.textContent = `${payload.runId || "benchmark"} / ${payload.elapsedMs || 0} ms`;
  if (els.benchmarkArtifacts) {
    const artifacts = payload.artifacts || {};
    els.benchmarkArtifacts.textContent = `${artifacts.json || "json pending"} / ${artifacts.csv || "csv pending"}`;
  }

  renderBenchmarkGroups(summary.byGroup || {});
  renderBenchmarkCases(Array.isArray(payload.results) ? payload.results : []);
}

function renderBenchmarkGroups(groups) {
  if (!els.benchmarkGroupRows) return;
  const entries = Object.entries(groups);
  if (!entries.length) {
    els.benchmarkGroupRows.textContent = "No group metrics.";
    return;
  }
  els.benchmarkGroupRows.replaceChildren(
    ...entries.map(([name, row]) => {
      const item = document.createElement("article");
      item.className = `benchmark-group-row ${Number(row.passRate || 0) >= 1 ? "pass" : "review"}`;
      const title = document.createElement("strong");
      title.textContent = name;
      const score = document.createElement("span");
      score.textContent = `${row.passed || 0} / ${row.total || 0} passed / ${formatPercent(row.passRate)}`;
      item.append(title, score);
      return item;
    }),
  );
}

function renderBenchmarkCases(results) {
  if (!els.benchmarkCaseRows) return;
  if (!results.length) {
    els.benchmarkCaseRows.textContent = "No case results.";
    return;
  }
  els.benchmarkCaseRows.replaceChildren(
    ...results.map((result) => {
      const row = document.createElement("article");
      row.className = `benchmark-case-row ${result.passed ? "pass" : "fail"}`;
      const title = document.createElement("strong");
      title.textContent = `${result.passed ? "PASS" : "FAIL"} - ${result.id}`;
      const meta = document.createElement("span");
      meta.textContent = `${result.type} / ${result.group} / ${result.condition} / ${formatMs(result.latencyMs)} / voice ${formatMs(result.estimatedVoiceLatencyMs)} / ${formatMoney(result.cost?.vapiStackCost || 0)}`;
      const answer = document.createElement("p");
      answer.textContent = result.answer || result.query || "No answer.";
      const failures = document.createElement("span");
      failures.textContent = result.failures?.length ? `failures: ${result.failures.join(" | ")}` : "failures: none";
      row.append(title, meta, answer, failures);
      return row;
    }),
  );
}

function renderBenchmarkError(error) {
  if (els.benchmarkPassMetric) els.benchmarkPassMetric.textContent = "failed";
  if (els.benchmarkCaseRows) {
    const empty = document.createElement("div");
    empty.className = "trace-empty";
    empty.textContent = readableError(error);
    els.benchmarkCaseRows.replaceChildren(empty);
  }
  if (els.benchmarkGroupRows) els.benchmarkGroupRows.textContent = "Benchmark failed.";
}
async function loadSpeechCases() {
  if (!els.speechGroupSelect) return;
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/speech/cases"));
    if (!response.ok) throw new Error("Speech cases unavailable");
    const payload = await response.json();
    const groups = Array.isArray(payload.groups) ? payload.groups : [];
    const current = els.speechGroupSelect.value;
    els.speechGroupSelect.replaceChildren(
      optionNode("", "All groups"),
      ...groups.map((group) => optionNode(group, group)),
    );
    els.speechGroupSelect.value = groups.includes(current) ? current : "";
  } catch {
    els.speechGroupSelect.replaceChildren(optionNode("", "All groups"));
  }
}

async function loadLatestSpeechEval() {
  if (!els.speechCaseRows) return;
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/speech/latest"));
    if (!response.ok) throw new Error("Latest speech evaluation unavailable");
    const payload = await response.json();
    if (payload?.found === false) return;
    renderSpeechEval(payload);
  } catch {
    // The speech suite may not have been run yet.
  }
}

async function runSpeechSuite() {
  if (!els.speechRunButton) return;
  els.speechRunButton.disabled = true;
  els.speechRunButton.textContent = "Running";
  if (els.speechCaseRows) els.speechCaseRows.textContent = "Running speech robustness cases.";
  if (els.speechConditionRows) els.speechConditionRows.textContent = "Scoring speech conditions.";
  try {
    const limit = Number.parseInt(els.speechLimitInput?.value || "", 10);
    const group = els.speechGroupSelect?.value || "";
    const response = await fetch(previewBackendUrl("/api/evaluation/speech/run"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        groups: group ? [group] : [],
        conditions: [],
        limit: Number.isFinite(limit) && limit > 0 ? limit : null,
        save: true,
      }),
    });
    if (!response.ok) throw new Error("Speech robustness suite failed");
    const payload = await response.json();
    renderSpeechEval(payload);
    recordTraceEvent("speech_robustness_suite", {
      runId: payload.runId,
      total: payload.summary?.total,
      passed: payload.summary?.passed,
      passRate: payload.summary?.passRate,
      avgWer: payload.summary?.avgWer,
      entityRecall: payload.summary?.avgEntityRecall,
      voiceP95: payload.summary?.latency?.voiceP95Ms,
      per1000: payload.summary?.cost?.per1000VapiStack,
    });
  } catch (error) {
    renderSpeechError(error);
  } finally {
    els.speechRunButton.disabled = false;
    els.speechRunButton.textContent = "Run speech suite";
  }
}

function renderSpeechEval(payload) {
  state.latestSpeechEval = payload;
  const summary = payload?.summary || {};
  const latency = summary.latency || {};
  const cost = summary.cost || {};

  if (els.speechCasesMetric) els.speechCasesMetric.textContent = `${summary.passed ?? 0} / ${summary.total ?? 0}`;
  if (els.speechPassMetric) els.speechPassMetric.textContent = formatPercent(summary.passRate);
  if (els.speechWerMetric) els.speechWerMetric.textContent = ratioText(summary.avgWer);
  if (els.speechEntityMetric) els.speechEntityMetric.textContent = formatPercent(summary.avgEntityRecall);
  if (els.speechVoiceMetric) els.speechVoiceMetric.textContent = formatMs(latency.voiceP95Ms);
  if (els.speechCostMetric) els.speechCostMetric.textContent = formatMoney(cost.per1000VapiStack || 0);
  if (els.speechRunId) els.speechRunId.textContent = `${payload.runId || "speech-suite"} / ${payload.elapsedMs || 0} ms`;
  if (els.speechArtifacts) {
    const artifacts = payload.artifacts || {};
    els.speechArtifacts.textContent = `${artifacts.json || "json pending"} / ${artifacts.csv || "csv pending"}`;
  }

  renderSpeechConditions(summary.byCondition || {});
  renderSpeechCases(Array.isArray(payload.results) ? payload.results : []);
}

function renderSpeechConditions(conditions) {
  if (!els.speechConditionRows) return;
  const entries = Object.entries(conditions);
  if (!entries.length) {
    els.speechConditionRows.textContent = "No condition metrics.";
    return;
  }
  els.speechConditionRows.replaceChildren(
    ...entries.map(([name, row]) => {
      const item = document.createElement("article");
      item.className = `benchmark-group-row ${Number(row.passRate || 0) >= 1 ? "pass" : "review"}`;
      const title = document.createElement("strong");
      title.textContent = name;
      const score = document.createElement("span");
      score.textContent = `${row.passed || 0} / ${row.total || 0} passed / WER ${ratioText(row.avgWer)} / entity ${formatPercent(row.avgEntityRecall)}`;
      item.append(title, score);
      return item;
    }),
  );
}

function renderSpeechCases(results) {
  if (!els.speechCaseRows) return;
  if (!results.length) {
    els.speechCaseRows.textContent = "No speech case results.";
    return;
  }
  els.speechCaseRows.replaceChildren(
    ...results.map((result) => {
      const row = document.createElement("article");
      row.className = `benchmark-case-row ${result.passed ? "pass" : "fail"}`;
      const title = document.createElement("strong");
      title.textContent = `${result.passed ? "PASS" : "FAIL"} - ${result.id}`;
      const condition = result.condition || {};
      const meta = document.createElement("span");
      meta.textContent = `${result.route} / ${result.group} / ${condition.accent || "accent"} / ${condition.noise || "noise"} / WER ${ratioText(result.wer)} / entity ${formatPercent(result.entityRecall)} / voice ${formatMs(result.estimatedVoiceLatencyMs)} / ${formatMoney(result.cost?.vapiStackCost || 0)}`;
      const transcript = document.createElement("p");
      transcript.textContent = `${result.referenceText || "Reference"} -> ${result.transcriptText || "Transcript"}`;
      const answer = document.createElement("p");
      answer.textContent = result.answer || JSON.stringify(result.observed || {});
      const failures = document.createElement("span");
      failures.textContent = result.failures?.length ? `failures: ${result.failures.join(" | ")}` : "failures: none";
      row.append(title, meta, transcript, answer, failures);
      return row;
    }),
  );
}

function renderSpeechError(error) {
  if (els.speechPassMetric) els.speechPassMetric.textContent = "failed";
  if (els.speechCaseRows) {
    const empty = document.createElement("div");
    empty.className = "trace-empty";
    empty.textContent = readableError(error);
    els.speechCaseRows.replaceChildren(empty);
  }
  if (els.speechConditionRows) els.speechConditionRows.textContent = "Speech suite failed.";
}

function ratioText(value, digits = 3) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "-";
}
function audioTargetPerPrompt() {
  const value = Number.parseInt(els.audioTargetInput?.value || "3", 10);
  return Number.isFinite(value) && value > 0 ? value : 3;
}

function audioMetadataPayload() {
  const distance = Number.parseInt(els.audioDistanceInput?.value || "30", 10);
  return {
    speakerId: els.audioSpeakerInput?.value?.trim() || "speaker-1",
    accent: els.audioAccentSelect?.value || "user_recorded",
    noise: els.audioNoiseSelect?.value || "room",
    device: els.audioDeviceSelect?.value || "browser_mic",
    micDistanceCm: Number.isFinite(distance) ? distance : 30,
    notes: els.audioNotesInput?.value?.trim() || "",
  };
}
function selectedAudioTemplate() {
  const id = els.audioCaseSelect?.value || "";
  return (state.latestAudioCases?.templates || []).find((item) => item.id === id) || null;
}

function syncAudioReference() {
  const template = selectedAudioTemplate();
  if (els.audioReferenceInput && template) {
    els.audioReferenceInput.value = template.referenceText || "";
  }
}

function supportedAudioMimeType() {
  if (!window.MediaRecorder) return "";
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

async function toggleAudioRecording() {
  if (state.audioRecorder?.state === "recording") {
    stopAudioRecording();
    return;
  }
  await startAudioRecording();
}

async function startAudioRecording() {
  if (!els.audioRecordButton) return;
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    setAudioStatus("Browser recording is unavailable in this environment.");
    return;
  }
  try {
    state.audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.audioChunks = [];
    state.audioBlob = null;
    const mimeType = supportedAudioMimeType();
    const options = mimeType ? { mimeType } : undefined;
    state.audioRecorder = new MediaRecorder(state.audioStream, options);
    state.audioRecorder.ondataavailable = (event) => {
      if (event.data?.size) state.audioChunks.push(event.data);
    };
    state.audioRecorder.onstop = () => {
      const type = state.audioRecorder?.mimeType || mimeType || "audio/webm";
      state.audioBlob = new Blob(state.audioChunks, { type });
      if (els.audioPreview) els.audioPreview.src = URL.createObjectURL(state.audioBlob);
      if (els.audioSaveButton) els.audioSaveButton.disabled = false;
      setAudioStatus(`Captured ${Math.round(state.audioBlob.size / 1024)} KB recording.`);
      state.audioStream?.getTracks().forEach((track) => track.stop());
      state.audioStream = null;
      state.audioRecorder = null;
      renderAudioRecordButton(false);
    };
    state.audioRecordingStartedAt = performance.now();
    state.audioRecorder.start();
    if (els.audioSaveButton) els.audioSaveButton.disabled = true;
    renderAudioRecordButton(true);
    setAudioStatus("Recording audio fixture.");
  } catch (error) {
    setAudioStatus(readableError(error));
    renderAudioRecordButton(false);
  }
}

function stopAudioRecording() {
  if (state.audioRecorder?.state === "recording") {
    state.audioRecorder.stop();
  }
}

function renderAudioRecordButton(recording) {
  if (!els.audioRecordButton) return;
  els.audioRecordButton.classList.toggle("recording", recording);
  els.audioRecordButton.textContent = recording ? "Stop" : "Record";
}

function setAudioStatus(message) {
  if (els.audioStatus) els.audioStatus.textContent = message;
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error("Could not read audio blob"));
    reader.readAsDataURL(blob);
  });
}

async function saveAudioRecording() {
  const template = selectedAudioTemplate();
  if (!template || !state.audioBlob) {
    setAudioStatus("Select a prompt and record audio first.");
    return;
  }
  if (els.audioSaveButton) {
    els.audioSaveButton.disabled = true;
    els.audioSaveButton.textContent = "Saving";
  }
  try {
    const audioBase64 = await blobToDataUrl(state.audioBlob);
    const durationMs = state.audioRecordingStartedAt ? Math.round(performance.now() - state.audioRecordingStartedAt) : 0;
    const response = await fetch(previewBackendUrl("/api/evaluation/audio/recording"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        caseId: template.id,
        referenceText: els.audioReferenceInput?.value || template.referenceText || "",
        audioBase64,
        mimeType: state.audioBlob.type || "audio/webm",
        durationMs,
        metadata: audioMetadataPayload(),
      }),
    });
    if (!response.ok) throw new Error(await response.text());
    const payload = await response.json();
    state.audioBlob = null;
    if (els.audioPreview) els.audioPreview.removeAttribute("src");
    setAudioStatus(`Saved ${payload.case?.id || "recording"}.`);
    await loadAudioCases();
    await buildAudioManifest({ silent: true });
  } catch (error) {
    setAudioStatus(readableError(error));
    if (els.audioSaveButton) els.audioSaveButton.disabled = false;
  } finally {
    if (els.audioSaveButton) els.audioSaveButton.textContent = "Save recording";
  }
}

function selectedAudioRecording() {
  const recordings = Array.isArray(state.latestAudioCases?.recordings) ? state.latestAudioCases.recordings : [];
  const templateId = els.audioCaseSelect?.value || "";
  const newestFirst = recordings.slice().reverse();
  return (
    newestFirst.find((item) => item.templateId === templateId && !item.parentRecordingId) ||
    newestFirst.find((item) => item.templateId === templateId) ||
    newestFirst[0] ||
    null
  );
}

async function decodeRecordingAudio(recording) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) throw new Error("Browser audio decoding is unavailable.");
  const response = await fetch(previewBackendUrl(`/api/evaluation/audio/recordings/${encodeURIComponent(recording.id)}/file`), {
    cache: "no-store",
  });
  if (!response.ok) throw new Error(`Could not fetch ${recording.id} audio.`);
  const data = await response.arrayBuffer();
  const context = new AudioContextClass();
  try {
    return await context.decodeAudioData(data.slice(0));
  } finally {
    if (typeof context.close === "function") context.close();
  }
}

function deterministicNoise(index, channel) {
  const value = Math.sin((index + 1) * (channel + 1) * 12.9898) * 43758.5453;
  return (value - Math.floor(value)) * 2 - 1;
}

function stressVariantSamples(buffer, variant) {
  const sampleRate = buffer.sampleRate || 48000;
  const channels = Math.max(1, Math.min(2, buffer.numberOfChannels || 1));
  const playbackRate = Number.isFinite(variant.playbackRate) && variant.playbackRate > 0 ? variant.playbackRate : 1;
  const sourceLength = Math.max(1, buffer.length || 1);
  const targetLength = Math.max(1, Math.round(sourceLength / playbackRate));
  const speechGain = Number.isFinite(variant.speechGain) ? variant.speechGain : 1;
  const noiseAmplitude = Number.isFinite(variant.noiseAmplitude) ? variant.noiseAmplitude : 0;
  const beepAmplitude = Number.isFinite(variant.beepAmplitude) ? variant.beepAmplitude : 0;
  const clipAt = Number.isFinite(variant.clipAt) ? Math.max(0.1, Math.min(1, variant.clipAt)) : null;
  const output = [];

  for (let channel = 0; channel < channels; channel += 1) {
    const source = buffer.getChannelData(channel);
    const target = new Float32Array(targetLength);
    for (let index = 0; index < targetLength; index += 1) {
      const sourceIndex = Math.min(sourceLength - 1, index * playbackRate);
      const lower = Math.floor(sourceIndex);
      const upper = Math.min(sourceLength - 1, lower + 1);
      const weight = sourceIndex - lower;
      let sample = (source[lower] || 0) * (1 - weight) + (source[upper] || 0) * weight;
      sample *= speechGain;
      if (noiseAmplitude) sample += deterministicNoise(index, channel) * noiseAmplitude;
      if (beepAmplitude) {
        const windowSamples = Math.max(1, Math.round(sampleRate * 0.72));
        const beepSamples = Math.max(1, Math.round(sampleRate * 0.035));
        if (index % windowSamples < beepSamples) {
          sample += Math.sin((2 * Math.PI * 1760 * index) / sampleRate) * beepAmplitude;
        }
      }
      if (clipAt !== null) sample = Math.max(-clipAt, Math.min(clipAt, sample));
      target[index] = Math.max(-1, Math.min(1, sample));
    }
    output.push(target);
  }

  return {
    channels: output,
    sampleRate,
    durationMs: Math.round((targetLength / sampleRate) * 1000),
  };
}

function writeAscii(view, offset, text) {
  for (let index = 0; index < text.length; index += 1) {
    view.setUint8(offset + index, text.charCodeAt(index));
  }
}

function wavBlobFromSamples(samples) {
  const channels = samples.channels;
  const channelCount = channels.length;
  const sampleRate = samples.sampleRate;
  const sampleCount = channels[0]?.length || 0;
  const bytesPerSample = 2;
  const blockAlign = channelCount * bytesPerSample;
  const dataSize = sampleCount * blockAlign;
  const arrayBuffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(arrayBuffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channelCount, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let index = 0; index < sampleCount; index += 1) {
    for (let channel = 0; channel < channelCount; channel += 1) {
      const sample = Math.max(-1, Math.min(1, channels[channel][index] || 0));
      const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      view.setInt16(offset, intSample, true);
      offset += bytesPerSample;
    }
  }

  return new Blob([arrayBuffer], { type: "audio/wav" });
}

function stressVariantMetadata(recording, variant) {
  const base = recording.recordingMetadata || {};
  const condition = recording.condition || {};
  const variantMetadata = variant.metadata || {};
  const notes = [base.notes, `generated from ${recording.id}`, variant.label].filter(Boolean).join(" / ");
  return {
    speakerId: base.speakerId || recording.speaker?.id || "speaker-1",
    accent: base.accent || recording.speaker?.accent || condition.accent || "user_recorded",
    noise: variantMetadata.noise || base.noise || condition.noise || "synthetic_stress",
    device: variantMetadata.device || base.device || condition.device || "browser_mic_augmented",
    environment: variantMetadata.environment || base.environment || condition.environment || "audio_stress_lab",
    micDistanceCm: Number(base.micDistanceCm || condition.micDistanceCm || 30),
    notes,
  };
}

async function generateAudioStressVariants() {
  if (!els.audioStressButton) return;
  const recording = selectedAudioRecording();
  if (!recording) {
    setAudioStatus("Save at least one recording before generating stress variants.");
    return;
  }

  els.audioStressButton.disabled = true;
  els.audioStressButton.textContent = "Generating";
  setAudioStatus(`Generating ${audioStressVariants.length} stress variants from ${recording.id}.`);
  try {
    const audioBuffer = await decodeRecordingAudio(recording);
    const saved = [];
    for (const variant of audioStressVariants) {
      const samples = stressVariantSamples(audioBuffer, variant);
      const blob = wavBlobFromSamples(samples);
      const audioBase64 = await blobToDataUrl(blob);
      const response = await fetch(previewBackendUrl("/api/evaluation/audio/recording"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          caseId: recording.templateId || selectedAudioTemplate()?.id || recording.id,
          referenceText: recording.referenceText || els.audioReferenceInput?.value || "",
          audioBase64,
          mimeType: "audio/wav",
          durationMs: samples.durationMs,
          metadata: stressVariantMetadata(recording, variant),
          parentRecordingId: recording.id,
          variantOf: recording.variantOf || recording.id,
          augmentation: {
            type: variant.type,
            label: variant.label,
            parameters: {
              playbackRate: variant.playbackRate,
              speechGain: variant.speechGain,
              noiseAmplitude: variant.noiseAmplitude || 0,
              beepAmplitude: variant.beepAmplitude || 0,
              clipAt: variant.clipAt || null,
            },
          },
        }),
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      saved.push(payload.case?.id || variant.type);
    }
    setAudioStatus(`Generated ${saved.length} stress variants. Run audio suite to score them with the speech recognizer.`);
    await loadAudioCases();
    await buildAudioManifest({ silent: true });
  } catch (error) {
    setAudioStatus(readableError(error));
  } finally {
    els.audioStressButton.disabled = false;
    els.audioStressButton.textContent = "Generate stress variants";
  }
}
async function loadAudioCases() {
  if (!els.audioCaseSelect) return;
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/audio/cases"));
    if (!response.ok) throw new Error("Audio cases unavailable");
    const payload = await response.json();
    state.latestAudioCases = payload;
    const templates = Array.isArray(payload.templates) ? payload.templates : [];
    const current = els.audioCaseSelect.value;
    els.audioCaseSelect.replaceChildren(
      ...templates.map((item) => optionNode(item.id, item.referenceText || item.id)),
    );
    els.audioCaseSelect.value = templates.some((item) => item.id === current) ? current : templates[0]?.id || "";
    syncAudioReference();
    renderAudioRecordings(payload.recordings || []);
    if (els.audioRecordingsMetric) els.audioRecordingsMetric.textContent = String(payload.recordingCount ?? 0);
    if (!payload.provider?.ready) setAudioStatus("Speech evaluation key is not set on the private service.");
  } catch (error) {
    setAudioStatus(readableError(error));
    els.audioCaseSelect.replaceChildren(optionNode("", "Audio cases unavailable"));
  }
}

async function loadLatestAudioManifest() {
  if (!els.audioManifestRows) return;
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/audio/manifest/latest"));
    if (!response.ok) throw new Error("Audio manifest unavailable");
    const payload = await response.json();
    if (payload?.found === false) return;
    renderAudioManifest(payload);
  } catch {
    // The manifest may not have been built yet.
  }
}

async function buildAudioManifest({ silent = false } = {}) {
  if (!els.audioManifestButton) return;
  if (!silent) {
    els.audioManifestButton.disabled = true;
    els.audioManifestButton.textContent = "Building";
    if (els.audioManifestRows) els.audioManifestRows.textContent = "Building dataset coverage.";
  }
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/audio/manifest"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ targetPerPrompt: audioTargetPerPrompt(), save: true }),
    });
    if (!response.ok) throw new Error("Audio manifest failed");
    const payload = await response.json();
    renderAudioManifest(payload);
    return payload;
  } catch (error) {
    if (!silent) setAudioStatus(readableError(error));
    return null;
  } finally {
    if (!silent) {
      els.audioManifestButton.disabled = false;
      els.audioManifestButton.textContent = "Build manifest";
    }
  }
}

function renderAudioManifest(payload) {
  state.latestAudioManifest = payload;
  const summary = payload?.summary || {};
  if (els.audioCoverageMetric) els.audioCoverageMetric.textContent = formatPercent(summary.coverageRate);
  if (els.audioCompleteMetric) els.audioCompleteMetric.textContent = `${summary.completePrompts ?? 0} / ${summary.templateCount ?? 0}`;
  if (els.audioSpeakersMetric) els.audioSpeakersMetric.textContent = String(summary.speakerCount ?? 0);
  if (els.audioConditionsMetric) els.audioConditionsMetric.textContent = String(summary.conditionCount ?? 0);
  if (els.audioMissingMetric) els.audioMissingMetric.textContent = String(summary.missingRecordings ?? 0);
  if (els.audioManifestMetric) els.audioManifestMetric.textContent = `${summary.recordingCount ?? 0} / ${summary.requiredRecordings ?? 0}`;
  if (els.audioManifestArtifacts) {
    const artifacts = payload.artifacts || {};
    els.audioManifestArtifacts.textContent = `${artifacts.json || "manifest json pending"} / ${artifacts.csv || "manifest csv pending"}`;
  }
  renderAudioManifestRows(Array.isArray(payload.promptCoverage) ? payload.promptCoverage : []);
}

function renderAudioManifestRows(rows) {
  if (!els.audioManifestRows) return;
  if (!rows.length) {
    els.audioManifestRows.textContent = "No prompt coverage yet.";
    return;
  }
  els.audioManifestRows.replaceChildren(
    ...rows.map((row) => {
      const item = document.createElement("article");
      item.className = `benchmark-group-row ${row.complete ? "pass" : "review"}`;
      const title = document.createElement("strong");
      title.textContent = row.referenceText || row.templateId;
      const meta = document.createElement("span");
      meta.textContent = `${row.recordings || 0} / ${row.target || 0} recordings / missing ${row.missing || 0} / ${formatPercent(row.coverageRate)}`;
      item.append(title, meta);
      return item;
    }),
  );
}

async function loadLatestAudioQuality() {
  if (!els.audioQualityRows) return;
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/audio/quality/latest"));
    if (!response.ok) throw new Error("Latest audio QA unavailable");
    const payload = await response.json();
    if (payload?.found === false) return;
    renderAudioQuality(payload);
  } catch {
    // The audio QA gate may not have been run yet.
  }
}

async function runAudioQuality({ silent = false } = {}) {
  if (!els.audioQualityRows) return null;
  if (!silent && els.audioQualityButton) {
    els.audioQualityButton.disabled = true;
    els.audioQualityButton.textContent = "Building";
  }
  if (!silent) setAudioStatus("Building real-audio retake queue.");
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/audio/quality"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ save: true }),
    });
    if (!response.ok) throw new Error("Audio QA gate failed");
    const payload = await response.json();
    renderAudioQuality(payload);
    if (!silent) setAudioStatus(`Retake queue updated: ${payload.summary?.retakeNeeded ?? 0} retakes.`);
    return payload;
  } catch (error) {
    if (!silent) setAudioStatus(readableError(error));
    return null;
  } finally {
    if (!silent && els.audioQualityButton) {
      els.audioQualityButton.disabled = false;
      els.audioQualityButton.textContent = "Build retake queue";
    }
  }
}

function renderAudioQuality(payload) {
  state.latestAudioQuality = payload;
  const summary = payload?.summary || {};
  if (els.audioUsableMetric) els.audioUsableMetric.textContent = `${summary.usableForPaper ?? 0} / ${summary.totalRecordings ?? 0}`;
  if (els.audioRetakeMetric) els.audioRetakeMetric.textContent = String(summary.retakeNeeded ?? 0);
  if (els.audioUrgentMetric) els.audioUrgentMetric.textContent = String(summary.urgentRetakes ?? 0);
  if (els.audioEmptyMetric) els.audioEmptyMetric.textContent = String(summary.emptyTranscripts ?? 0);
  if (els.audioWrongPromptMetric) els.audioWrongPromptMetric.textContent = String(summary.wrongPromptOrUnintelligible ?? 0);
  if (els.audioQualityScoreMetric) els.audioQualityScoreMetric.textContent = ratioText(summary.avgQualityScore);
  if (els.audioQualityArtifacts) {
    const artifacts = payload.artifacts || {};
    els.audioQualityArtifacts.textContent = `${artifacts.json || "qa json pending"} / ${artifacts.retakeQueueJson || "retake queue pending"}`;
  }
  renderAudioQualityRows(payload.retakeQueue || []);
}

function renderAudioQualityRows(rows) {
  if (!els.audioQualityRows) return;
  if (!rows.length) {
    els.audioQualityRows.textContent = "No retakes needed.";
    return;
  }
  els.audioQualityRows.replaceChildren(
    ...rows.slice(0, 18).map((row) => {
      const item = document.createElement("article");
      item.className = `benchmark-case-row ${row.priority === "urgent_retake" ? "fail" : "review"}`;
      const title = document.createElement("strong");
      title.textContent = `${row.priority || "review"} / score ${row.qualityScore ?? "-"} - ${row.id || "-"}`;
      const transcript = document.createElement("span");
      transcript.textContent = `${row.referenceText || "-"} -> ${row.transcriptText || "No transcript"}`;
      const modes = document.createElement("span");
      modes.textContent = Array.isArray(row.failureModes) ? row.failureModes.join(" | ") : "-";
      const action = document.createElement("span");
      action.textContent = row.retakeInstruction || "-";
      item.append(title, transcript, modes, action);
      return item;
    }),
  );
}

async function loadLatestAudioAcceptedSet() {
  if (!els.audioAcceptedRows) return;
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/audio/accepted-set/latest"));
    if (!response.ok) throw new Error("Latest accepted set unavailable");
    const payload = await response.json();
    if (payload?.found === false) return;
    renderAudioAcceptedSet(payload);
  } catch {
    // The accepted-set builder may not have been run yet.
  }
}

async function runAudioAcceptedSet({ silent = false } = {}) {
  if (!els.audioAcceptedRows) return null;
  if (!silent && els.audioAcceptedButton) {
    els.audioAcceptedButton.disabled = true;
    els.audioAcceptedButton.textContent = "Building";
  }
  if (!silent) setAudioStatus("Building accepted real-audio benchmark set.");
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/audio/accepted-set"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ save: true }),
    });
    if (!response.ok) throw new Error("Accepted-set build failed");
    const payload = await response.json();
    renderAudioAcceptedSet(payload);
    if (!silent) setAudioStatus(`Accepted set updated: ${payload.summary?.acceptedRecordings ?? 0} fixtures.`);
    return payload;
  } catch (error) {
    if (!silent) setAudioStatus(readableError(error));
    return null;
  } finally {
    if (!silent && els.audioAcceptedButton) {
      els.audioAcceptedButton.disabled = false;
      els.audioAcceptedButton.textContent = "Build accepted set";
    }
  }
}

function renderAudioAcceptedSet(payload) {
  state.latestAudioAcceptedSet = payload;
  const summary = payload?.summary || {};
  if (els.audioAcceptedMetric) els.audioAcceptedMetric.textContent = `${summary.acceptedRecordings ?? 0} / ${summary.groupCount ?? 0}`;
  if (els.audioAcceptedPassMetric) els.audioAcceptedPassMetric.textContent = formatPercent(summary.acceptedPassRate);
  if (els.audioSupersededMetric) els.audioSupersededMetric.textContent = String(summary.supersededRecordings ?? 0);
  if (els.audioAcceptedRetakeMetric) els.audioAcceptedRetakeMetric.textContent = String(summary.groupsNeedingRetake ?? 0);
  if (els.audioAcceptedWerMetric) els.audioAcceptedWerMetric.textContent = ratioText(summary.acceptedAvgWer);
  if (els.audioAcceptedLiftMetric) els.audioAcceptedLiftMetric.textContent = formatSignedPercentDelta(summary.passRateLiftVsRaw);
  if (els.audioAcceptedArtifacts) {
    const artifacts = payload.artifacts || {};
    els.audioAcceptedArtifacts.textContent = `${artifacts.json || "accepted json pending"} / ${artifacts.csv || "accepted csv pending"}`;
  }
  renderAudioAcceptedRows(payload);
}

function renderAudioAcceptedRows(payload) {
  if (!els.audioAcceptedRows) return;
  const rows = Array.isArray(payload?.rows) ? payload.rows : [];
  if (!rows.length) {
    els.audioAcceptedRows.textContent = "Run the accepted-set builder after an audio suite run.";
    return;
  }
  const statusOrder = { accepted: 0, rejected_retake: 1, needs_retake: 2, unevaluated: 3, superseded: 4, archived_attempt: 5 };
  const ordered = rows
    .slice()
    .sort((left, right) => (statusOrder[left.reviewStatus] ?? 9) - (statusOrder[right.reviewStatus] ?? 9))
    .slice(0, 22);
  els.audioAcceptedRows.replaceChildren(
    ...ordered.map((row) => {
      const item = document.createElement("article");
      item.className = `benchmark-case-row ${row.accepted ? "pass" : row.reviewStatus === "superseded" ? "review" : "fail"}`;
      const title = document.createElement("strong");
      title.textContent = `${row.reviewStatus || "review"} - ${row.recordingId || "-"}`;
      const metrics = document.createElement("span");
      const supersedes = Array.isArray(row.supersedes) && row.supersedes.length ? ` / supersedes ${row.supersedes.length}` : "";
      const supersededBy = row.supersededBy ? ` / superseded by ${compactId(row.supersededBy)}` : "";
      metrics.textContent = `${row.accent || "accent"} / ${row.noise || "noise"} / WER ${ratioText(row.wer)} / entity ${formatPercent(row.entityRecall)}${supersedes}${supersededBy}`;
      const transcript = document.createElement("span");
      transcript.textContent = `${row.referenceText || "-"} -> ${row.transcriptText || "No transcript"}`;
      const reason = document.createElement("span");
      reason.textContent = row.selectionReason || "-";
      item.append(title, metrics, transcript, reason);
      return item;
    }),
  );
}

async function loadLatestAudioErrorAnalysis() {
  if (!els.audioErrorRows) return;
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/audio/error-analysis/latest"));
    if (!response.ok) throw new Error("Latest audio error analysis unavailable");
    const payload = await response.json();
    if (payload?.found === false) return;
    renderAudioErrorAnalysis(payload);
  } catch {
    // The audio error analysis may not have been run yet.
  }
}

async function runAudioErrorAnalysis({ silent = false } = {}) {
  if (!els.audioErrorRows) return null;
  if (!silent && els.audioErrorButton) {
    els.audioErrorButton.disabled = true;
    els.audioErrorButton.textContent = "Analyzing";
  }
  if (!silent) setAudioStatus("Building real-audio error taxonomy.");
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/audio/error-analysis"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ save: true }),
    });
    if (!response.ok) throw new Error("Audio error analysis failed");
    const payload = await response.json();
    renderAudioErrorAnalysis(payload);
    if (!silent) setAudioStatus(`Error taxonomy updated: ${payload.summary?.failureBucketCount ?? 0} failure buckets.`);
    return payload;
  } catch (error) {
    if (!silent) setAudioStatus(readableError(error));
    return null;
  } finally {
    if (!silent && els.audioErrorButton) {
      els.audioErrorButton.disabled = false;
      els.audioErrorButton.textContent = "Analyze errors";
    }
  }
}

function renderAudioErrorAnalysis(payload) {
  state.latestAudioErrorAnalysis = payload;
  const summary = payload?.summary || {};
  if (els.audioErrorTopMetric) els.audioErrorTopMetric.textContent = summary.topFailure || "-";
  if (els.audioErrorBucketMetric) els.audioErrorBucketMetric.textContent = String(summary.failureBucketCount ?? 0);
  if (els.audioAsrOnlyMetric) els.audioAsrOnlyMetric.textContent = String(summary.asrOnlyFailures ?? 0);
  if (els.audioDownstreamFailMetric) els.audioDownstreamFailMetric.textContent = String(summary.downstreamFailures ?? 0);
  if (els.audioLanguageMismatchMetric) els.audioLanguageMismatchMetric.textContent = String(summary.languageMismatches ?? 0);
  if (els.audioCoverageGapMetric) els.audioCoverageGapMetric.textContent = String(summary.coverageGaps ?? 0);
  if (els.audioErrorArtifacts) {
    const artifacts = payload.artifacts || {};
    els.audioErrorArtifacts.textContent = `${artifacts.json || "error json pending"} / ${artifacts.actionPlanCsv || "action csv pending"}`;
  }
  renderAudioErrorRows(payload);
}

function renderAudioErrorRows(payload) {
  if (!els.audioErrorRows) return;
  const actions = Array.isArray(payload?.actionPlan) ? payload.actionPlan : [];
  const conditions = Array.isArray(payload?.conditionRisks) ? payload.conditionRisks : [];
  const failedRows = Array.isArray(payload?.rows)
    ? payload.rows.filter((row) => row.primaryFailure && row.primaryFailure !== "pass")
    : [];
  if (!actions.length && !failedRows.length) {
    els.audioErrorRows.textContent = "No blocking audio failures detected.";
    return;
  }

  const severityClass = (severity) => {
    if (severity === "blocker" || severity === "high") return "fail";
    if (severity === "medium" || severity === "low") return "review";
    return "pass";
  };

  const actionRows = actions.slice(0, 10).map((row) => {
    const item = document.createElement("article");
    item.className = `benchmark-case-row ${severityClass(row.severity)}`;
    const title = document.createElement("strong");
    title.textContent = `${row.failure || "failure"} / ${row.affectedRecordings ?? 0} recordings`;
    const meta = document.createElement("span");
    meta.textContent = `${row.family || "family"} / ${row.severity || "severity"} / ${row.paperAction || "paper action"}`;
    const action = document.createElement("span");
    action.textContent = row.recommendedAction || "-";
    item.append(title, meta, action);
    return item;
  });

  const conditionRows = conditions.slice(0, 6).map((row) => {
    const item = document.createElement("article");
    item.className = `benchmark-case-row ${Number(row.failureRate || 0) > 0.5 ? "fail" : "review"}`;
    const title = document.createElement("strong");
    title.textContent = `Condition ${row.condition || "-"} / ${formatPercent(row.failureRate)} failures`;
    const meta = document.createElement("span");
    meta.textContent = `top ${row.topFailure || "-"} / failed ${row.failed ?? 0} of ${row.total ?? 0} / WER ${ratioText(row.avgWer)} / entity ${formatPercent(row.avgEntityRecall)}`;
    item.append(title, meta);
    return item;
  });

  const examples = failedRows.slice(0, 8).map((row) => {
    const item = document.createElement("article");
    item.className = `benchmark-case-row ${severityClass(row.severity)}`;
    const title = document.createElement("strong");
    title.textContent = `${row.primaryFailure || "failure"} - ${compactId(row.id)}`;
    const meta = document.createElement("span");
    meta.textContent = `${row.accent || "accent"} / ${row.route || "route"} / WER ${ratioText(row.wer)} / entity ${formatPercent(row.entityRecall)}`;
    const transcript = document.createElement("span");
    transcript.textContent = `${row.referenceText || "-"} -> ${row.transcriptText || "No transcript"}`;
    const root = document.createElement("span");
    root.textContent = row.rootCause || "-";
    item.append(title, meta, transcript, root);
    return item;
  });

  els.audioErrorRows.replaceChildren(...actionRows, ...conditionRows, ...examples);
}

async function loadLatestAudioAccentSweep() {
  if (!els.audioAccentRows) return;
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/audio/accent-sweep/latest"));
    if (!response.ok) throw new Error("Latest accent sweep unavailable");
    const payload = await response.json();
    if (payload?.found === false) return;
    renderAudioAccentSweep(payload);
  } catch {
    // The accent sweep may not have been run yet.
  }
}

async function runAudioAccentSweep() {
  if (!els.audioAccentSweepButton) return null;
  els.audioAccentSweepButton.disabled = true;
  els.audioAccentSweepButton.textContent = "Sweeping";
  if (els.audioAccentRows) els.audioAccentRows.textContent = "Running accent configuration sweep.";
  setAudioStatus("Running accent-aware ASR sweep on failed/accented recordings.");
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/audio/accent-sweep"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        caseIds: [],
        limit: 8,
        includePassed: false,
        configs: [],
        allowReferenceFallback: Boolean(els.audioFallbackInput?.checked),
        save: true,
      }),
    });
    if (!response.ok) throw new Error("Accent sweep failed");
    const payload = await response.json();
    renderAudioAccentSweep(payload);
    setAudioStatus(`Accent sweep complete: best config ${payload.summary?.bestConfigLabel || "-"}.`);
    recordTraceEvent("audio_accent_sweep", {
      runId: payload.runId,
      bestConfigId: payload.summary?.bestConfigId,
      passRateLift: payload.summary?.passRateLift,
      entityRecallLift: payload.summary?.entityRecallLift,
    });
    return payload;
  } catch (error) {
    setAudioStatus(readableError(error));
    if (els.audioAccentRows) {
      const empty = document.createElement("div");
      empty.className = "trace-empty";
      empty.textContent = readableError(error);
      els.audioAccentRows.replaceChildren(empty);
    }
    return null;
  } finally {
    els.audioAccentSweepButton.disabled = false;
    els.audioAccentSweepButton.textContent = "Accent sweep";
  }
}

function formatSignedPercentDelta(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  const prefix = number > 0 ? "+" : "";
  return `${prefix}${Math.round(number * 100)}%`;
}

function renderAudioAccentSweep(payload) {
  state.latestAudioAccentSweep = payload;
  const summary = payload?.summary || {};
  if (els.audioAccentBestMetric) els.audioAccentBestMetric.textContent = summary.bestConfigLabel || "-";
  if (els.audioAccentLiftMetric) els.audioAccentLiftMetric.textContent = formatSignedPercentDelta(summary.passRateLift);
  if (els.audioAccentEntityMetric) els.audioAccentEntityMetric.textContent = formatSignedPercentDelta(summary.entityRecallLift);
  if (els.audioAccentCasesMetric) els.audioAccentCasesMetric.textContent = `${summary.caseCount ?? 0} x ${summary.configCount ?? 0}`;
  if (els.audioAccentArtifacts) {
    const artifacts = payload.artifacts || {};
    els.audioAccentArtifacts.textContent = `${artifacts.json || "accent json pending"} / ${artifacts.csv || "accent csv pending"}`;
  }
  renderAudioAccentRows(payload);
}

function renderAudioAccentRows(payload) {
  if (!els.audioAccentRows) return;
  const configs = Array.isArray(payload?.configSummaries) ? payload.configSummaries : [];
  const recommendations = Array.isArray(payload?.recommendations) ? payload.recommendations : [];
  if (!configs.length && !recommendations.length) {
    els.audioAccentRows.textContent = "Run an accent sweep after recording accented or failed clips.";
    return;
  }
  const bestId = payload?.summary?.bestConfigId;
  const configRows = configs.map((config) => {
    const item = document.createElement("article");
    item.className = `benchmark-case-row ${config.configId === bestId ? "pass" : "review"}`;
    const title = document.createElement("strong");
    title.textContent = `${config.configId === bestId ? "BEST" : "TEST"} - ${config.label || config.configId}`;
    const meta = document.createElement("span");
    meta.textContent = `pass ${formatPercent(config.passRate)} / WER ${ratioText(config.avgRawWer)} -> ${ratioText(config.avgWer)} / entity ${formatPercent(config.avgRawEntityRecall)} -> ${formatPercent(config.avgEntityRecall)} / p95 ${formatMs(config.deepgramP95Ms)} / repairs ${config.repairCount ?? 0}`;
    item.append(title, meta);
    return item;
  });
  const noteRows = recommendations.map((message) => {
    const item = document.createElement("article");
    item.className = "benchmark-case-row review";
    const title = document.createElement("strong");
    title.textContent = "Finding";
    const meta = document.createElement("span");
    meta.textContent = message;
    item.append(title, meta);
    return item;
  });
  els.audioAccentRows.replaceChildren(...configRows, ...noteRows);
}

function formatSignedRatio(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  const prefix = number > 0 ? "+" : "";
  return `${prefix}${number.toFixed(3)}`;
}

async function loadLatestAudioRobustness() {
  if (!els.audioRobustnessRows) return;
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/audio/robustness/latest"));
    if (!response.ok) throw new Error("Latest audio robustness unavailable");
    const payload = await response.json();
    if (payload?.found === false) return;
    renderAudioRobustness(payload);
  } catch {
    // The robustness analyzer may not have been run yet.
  }
}

async function runAudioRobustness({ silent = false } = {}) {
  if (!els.audioRobustnessRows) return null;
  if (!silent && els.audioRobustnessButton) {
    els.audioRobustnessButton.disabled = true;
    els.audioRobustnessButton.textContent = "Analyzing";
  }
  if (!silent) setAudioStatus("Analyzing audio robustness deltas.");
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/audio/robustness"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ save: true }),
    });
    if (!response.ok) throw new Error("Audio robustness analysis failed");
    const payload = await response.json();
    renderAudioRobustness(payload);
    if (!silent) setAudioStatus("Audio robustness analysis updated.");
    return payload;
  } catch (error) {
    if (!silent) setAudioStatus(readableError(error));
    return null;
  } finally {
    if (!silent && els.audioRobustnessButton) {
      els.audioRobustnessButton.disabled = false;
      els.audioRobustnessButton.textContent = "Analyze robustness";
    }
  }
}

function renderAudioRobustness(payload) {
  state.latestAudioRobustness = payload;
  const summary = payload?.summary || {};
  const worst = summary.worstVariant || null;
  if (els.audioRobustnessMetric) els.audioRobustnessMetric.textContent = `${summary.comparedCount ?? 0} / ${summary.variantCount ?? 0}`;
  if (els.audioRegressionMetric) els.audioRegressionMetric.textContent = summary.comparedCount ? `${summary.regressionCount ?? 0} (${formatPercent(summary.regressionRate)})` : "-";
  if (els.audioWorstVariantMetric) els.audioWorstVariantMetric.textContent = worst?.augmentationLabel || "-";
  if (els.audioWorstWerMetric) els.audioWorstWerMetric.textContent = formatSignedRatio(worst?.deltaWer);
  if (els.audioRobustnessArtifacts) {
    const artifacts = payload.artifacts || {};
    els.audioRobustnessArtifacts.textContent = `${artifacts.json || "robustness json pending"} / ${artifacts.csv || "robustness csv pending"}`;
  }
  renderAudioRobustnessRows(payload);
}

function renderAudioRobustnessRows(payload) {
  if (!els.audioRobustnessRows) return;
  const groups = Array.isArray(payload?.byAugmentation) ? payload.byAugmentation : [];
  const recommendations = Array.isArray(payload?.recommendations) ? payload.recommendations : [];
  if (!groups.length && !recommendations.length) {
    els.audioRobustnessRows.textContent = "Generate stress variants, run the audio suite, then analyze robustness.";
    return;
  }

  const groupRows = groups.map((group) => {
    const item = document.createElement("article");
    const regressionRate = Number(group.regressionRate || 0);
    item.className = `benchmark-case-row ${regressionRate > 0 ? "review" : "pass"}`;
    const title = document.createElement("strong");
    title.textContent = `${group.augmentationLabel || group.augmentationType}: ${formatPercent(group.regressionRate)} regressions`;
    const meta = document.createElement("span");
    meta.textContent = `compared ${group.comparedCount || 0} / pass ${formatPercent(group.passRate)} vs baseline ${formatPercent(group.baselinePassRate)} / delta WER ${formatSignedRatio(group.avgDeltaWer)} / delta entity ${formatSignedRatio(group.avgDeltaEntityRecall)} / ASR ${formatMs(group.avgDeltaTranscriptionLatencyMs)}`;
    item.append(title, meta);
    return item;
  });

  const noteRows = recommendations.map((message) => {
    const item = document.createElement("article");
    item.className = "benchmark-case-row review";
    const title = document.createElement("strong");
    title.textContent = "Finding";
    const meta = document.createElement("span");
    meta.textContent = message;
    item.append(title, meta);
    return item;
  });

  els.audioRobustnessRows.replaceChildren(...groupRows, ...noteRows);
}
async function loadLatestAudioEval() {
  if (!els.audioCaseRows) return;
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/audio/latest"));
    if (!response.ok) throw new Error("Latest audio evaluation unavailable");
    const payload = await response.json();
    if (payload?.found === false) return;
    renderAudioEval(payload);
  } catch {
    // The real audio suite may not have been run yet.
  }
}

async function runAudioSuite() {
  if (!els.audioRunButton) return;
  els.audioRunButton.disabled = true;
  els.audioRunButton.textContent = "Running";
  if (els.audioCaseRows) els.audioCaseRows.textContent = "Running real audio evaluation.";
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/audio/run"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        caseIds: [],
        limit: null,
        allowReferenceFallback: Boolean(els.audioFallbackInput?.checked),
        deepgramConfig: { id: "accent_aware_keyterms_repair" },
        enableTranscriptRepair: true,
        save: true,
      }),
    });
    if (!response.ok) throw new Error("Real audio evaluation failed");
    const payload = await response.json();
    renderAudioEval(payload);
    await buildAudioManifest({ silent: true });
    await runAudioQuality({ silent: true });
    await runAudioAcceptedSet({ silent: true });
    await runAudioErrorAnalysis({ silent: true });
    await runAudioRobustness({ silent: true });
    try {
      const reportResponse = await fetch(previewBackendUrl("/api/evaluation/report/run"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rerunSuites: false, save: true }),
      });
      if (reportResponse.ok) renderPaperReport(await reportResponse.json());
    } catch {
      // The audio run is still valid if report refresh fails.
    }
    recordTraceEvent("real_audio_eval", {
      runId: payload.runId,
      evaluated: payload.summary?.evaluated,
      passed: payload.summary?.passed,
      passRate: payload.summary?.passRate,
      wer: payload.summary?.avgWer,
      deepgramP95Ms: payload.summary?.latency?.deepgramP95Ms,
    });
  } catch (error) {
    renderAudioError(error);
  } finally {
    els.audioRunButton.disabled = false;
    els.audioRunButton.textContent = "Run audio suite";
  }
}

function renderAudioEval(payload) {
  state.latestAudioEval = payload;
  const summary = payload?.summary || {};
  const latency = summary.latency || {};
  if (els.audioEvaluatedMetric) els.audioEvaluatedMetric.textContent = `${summary.evaluated ?? 0} / ${summary.total ?? 0}`;
  if (els.audioPassMetric) els.audioPassMetric.textContent = summary.evaluated ? formatPercent(summary.passRate) : "-";
  if (els.audioWerMetric) els.audioWerMetric.textContent = ratioText(summary.avgWer);
  if (els.audioEntityMetric) els.audioEntityMetric.textContent = formatPercent(summary.avgEntityRecall);
  if (els.audioDeepgramMetric) els.audioDeepgramMetric.textContent = formatMs(latency.deepgramP95Ms);
  if (els.audioMultilingualMetric) els.audioMultilingualMetric.textContent = String(summary.multilingualScored ?? 0);
  if (els.audioSpanishPassMetric) els.audioSpanishPassMetric.textContent = formatPercent(summary.byLanguage?.es?.passRate);
  if (els.audioSurfaceWerMetric) els.audioSurfaceWerMetric.textContent = ratioText(summary.avgSurfaceWer);
  if (els.audioCanonicalWerMetric) els.audioCanonicalWerMetric.textContent = ratioText(summary.avgCanonicalWer);
  if (els.audioCanonicalEntityMetric) els.audioCanonicalEntityMetric.textContent = formatPercent(summary.avgCanonicalEntityRecall);
  if (els.audioDownstreamMetric) els.audioDownstreamMetric.textContent = formatPercent(summary.downstreamTaskSuccess);
  if (els.audioSemanticPassMetric) els.audioSemanticPassMetric.textContent = formatPercent(summary.semanticTranscriptPassRate);
  if (els.audioSemanticRecoveredMetric) els.audioSemanticRecoveredMetric.textContent = String(summary.semanticRecoveredAsrMisses ?? 0);
  if (els.audioSemanticScoreMetric) els.audioSemanticScoreMetric.textContent = ratioText(summary.avgSemanticScore);
  if (els.audioSemanticIntentMetric) els.audioSemanticIntentMetric.textContent = ratioText(summary.avgSemanticIntentScore);
  if (els.audioSemanticSlotMetric) els.audioSemanticSlotMetric.textContent = ratioText(summary.avgSemanticSlotScore);
  if (els.audioSemanticCanonicalMetric) els.audioSemanticCanonicalMetric.textContent = ratioText(summary.avgSemanticCanonicalScore);
  if (els.audioRunId) els.audioRunId.textContent = `${payload.runId || "audio-suite"} / ${payload.elapsedMs || 0} ms`;
  if (els.audioArtifacts) {
    const artifacts = payload.artifacts || {};
    els.audioArtifacts.textContent = `${artifacts.json || "json pending"} / ${artifacts.csv || "csv pending"}`;
  }
  renderAudioResults(Array.isArray(payload.results) ? payload.results : []);
}

function renderAudioRecordings(recordings) {
  if (!els.audioRecordingRows) return;
  if (!recordings.length) {
    els.audioRecordingRows.textContent = "No recordings yet.";
    return;
  }
  els.audioRecordingRows.replaceChildren(
    ...recordings.slice().reverse().map((item) => {
      const row = document.createElement("article");
      const augmentation = item.augmentation || {};
      row.className = `benchmark-group-row ${item.parentRecordingId ? "review" : "pass"}`;
      const title = document.createElement("strong");
      title.textContent = `${item.referenceText || item.id}${augmentation.label ? ` (${augmentation.label})` : ""}`;
      const meta = document.createElement("span");
      const metadata = item.recordingMetadata || {};
      const parent = item.parentRecordingId ? ` / parent ${compactId(item.parentRecordingId)}` : "";
      const variant = augmentation.type ? ` / ${augmentation.type}` : "";
      meta.textContent = `${item.id}${parent}${variant} / ${metadata.speakerId || "speaker"} / ${metadata.accent || (item.condition || {}).accent || "accent"} / ${metadata.noise || (item.condition || {}).noise || "noise"} / ${metadata.device || "device"} / ${formatMs(item.durationMs)}`;
      row.append(title, meta);
      return row;
    }),
  );
}

function renderAudioResults(results) {
  if (!els.audioCaseRows) return;
  if (!results.length) {
    els.audioCaseRows.textContent = "No audio case results yet.";
    return;
  }
  els.audioCaseRows.replaceChildren(
    ...results.map((result) => {
      const skipped = Boolean(result.skipped);
      const row = document.createElement("article");
      row.className = `benchmark-case-row ${skipped ? "review" : result.passed ? "pass" : "fail"}`;
      const title = document.createElement("strong");
      title.textContent = `${skipped ? "SKIP" : result.passed ? "PASS" : "FAIL"} - ${result.id}`;
      const meta = document.createElement("span");
      const augmentation = result.augmentation || {};
      const variant = augmentation.label ? `${augmentation.label} / ` : "";
      const rawMetric =
        result.transcriptRepairEnabled && result.rawWer !== undefined
          ? ` / raw WER ${ratioText(result.rawWer)} / raw entity ${formatPercent(result.rawEntityRecall)}`
          : "";
      const semantic = result.semanticTranscript || {};
      const semanticMetric = semantic.label ? ` / semantic ${semantic.label} ${formatPercent(semantic.score)}` : "";
      meta.textContent = skipped
        ? `${variant}${result.skipReason || "skipped"} / ${result.providerMessage || ""}`
        : `${variant}${result.transcriptionProvider || "provider"} / WER ${ratioText(result.wer)} / entity ${formatPercent(result.entityRecall)}${semanticMetric}${rawMetric} / ASR ${formatMs(result.transcriptionLatencyMs)}`;
      const transcript = document.createElement("p");
      const canonical =
        result.multilingualScoring && result.canonicalTranscriptText
          ? ` -> canonical: ${result.canonicalTranscriptText}`
          : "";
      transcript.textContent = `${result.referenceText || "Reference"} -> ${result.transcriptText || "No transcript"}${canonical}`;
      const answer = document.createElement("p");
      answer.textContent = result.answer || "No downstream answer.";
      const failures = document.createElement("span");
      failures.textContent = result.failures?.length ? `failures: ${result.failures.join(" | ")}` : "failures: none";
      row.append(title, meta, transcript, answer, failures);
      return row;
    }),
  );
}

function renderAudioError(error) {
  if (els.audioPassMetric) els.audioPassMetric.textContent = "failed";
  if (els.audioCaseRows) {
    const empty = document.createElement("div");
    empty.className = "trace-empty";
    empty.textContent = readableError(error);
    els.audioCaseRows.replaceChildren(empty);
  }
}
async function loadLatestPaperReport() {
  if (!els.reportCoreRows) return;
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/report/latest"));
    if (!response.ok) throw new Error("Latest paper report unavailable");
    const payload = await response.json();
    if (payload?.found === false) return;
    renderPaperReport(payload);
  } catch {
    // The report may not have been generated yet.
  }
}

async function loadLatestStatisticsPack() {
  if (!els.statsRows) return;
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/statistics/latest"));
    if (!response.ok) throw new Error("Latest statistics pack unavailable");
    const payload = await response.json();
    if (payload?.found === false) return;
    renderStatisticsPack(payload);
  } catch {
    // The statistics pack may not have been generated yet.
  }
}

async function runStatisticsPack({ silent = false } = {}) {
  if (!els.statsRows) return null;
  if (!silent && els.statsRunButton) {
    els.statsRunButton.disabled = true;
    els.statsRunButton.textContent = "Running";
  }
  if (!silent) {
    if (els.statsRows) els.statsRows.textContent = "Calculating confidence intervals.";
    if (els.statsArtifacts) els.statsArtifacts.textContent = "Statistics running.";
  }
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/statistics/run"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ iterations: 1000, confidence: 0.95, save: true }),
    });
    if (!response.ok) throw new Error("Statistics pack failed");
    const payload = await response.json();
    renderStatisticsPack(payload);
    recordTraceEvent("paper_statistics_pack", {
      runId: payload.runId,
      populatedMetricCount: payload.summary?.populatedMetricCount,
      metricCount: payload.summary?.metricCount,
    });
    return payload;
  } catch (error) {
    if (!silent) renderStatisticsError(error);
    return null;
  } finally {
    if (!silent && els.statsRunButton) {
      els.statsRunButton.disabled = false;
      els.statsRunButton.textContent = "Run stats";
    }
  }
}

async function loadLatestClaimReadiness() {
  if (!els.claimsRows) return;
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/claims/latest"));
    if (!response.ok) throw new Error("Latest claim readiness unavailable");
    const payload = await response.json();
    if (payload?.found === false) return;
    renderClaimReadiness(payload);
  } catch {
    // The claim readiness pack may not have been generated yet.
  }
}

async function runClaimReadiness({ silent = false } = {}) {
  if (!els.claimsRows) return null;
  if (!silent && els.claimsRunButton) {
    els.claimsRunButton.disabled = true;
    els.claimsRunButton.textContent = "Running";
  }
  if (!silent) {
    if (els.claimsRows) els.claimsRows.textContent = "Checking paper claims.";
    if (els.claimsArtifacts) els.claimsArtifacts.textContent = "Claims running.";
  }
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/claims/run"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ regenerateStatistics: false, save: true }),
    });
    if (!response.ok) throw new Error("Claim readiness failed");
    const payload = await response.json();
    renderClaimReadiness(payload);
    recordTraceEvent("paper_claim_readiness", {
      runId: payload.runId,
      publishable: payload.summary?.publishable,
      totalClaims: payload.summary?.totalClaims,
      claimReadinessScore: payload.summary?.claimReadinessScore,
    });
    return payload;
  } catch (error) {
    if (!silent) renderClaimReadinessError(error);
    return null;
  } finally {
    if (!silent && els.claimsRunButton) {
      els.claimsRunButton.disabled = false;
      els.claimsRunButton.textContent = "Run claims";
    }
  }
}

async function loadLatestExperimentPlan() {
  if (!els.planRows) return;
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/experiment-plan/latest"));
    if (!response.ok) throw new Error("Latest experiment plan unavailable");
    const payload = await response.json();
    if (payload?.found === false) return;
    renderExperimentPlan(payload);
  } catch {
    // The experiment plan may not have been generated yet.
  }
}

async function runExperimentPlan({ silent = false } = {}) {
  if (!els.planRows) return null;
  if (!silent && els.planRunButton) {
    els.planRunButton.disabled = true;
    els.planRunButton.textContent = "Planning";
  }
  if (!silent) {
    if (els.planRows) els.planRows.textContent = "Building experiment plan.";
    if (els.planArtifacts) els.planArtifacts.textContent = "Plan running.";
  }
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/experiment-plan/run"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refreshClaims: false, save: true }),
    });
    if (!response.ok) throw new Error("Experiment plan failed");
    const payload = await response.json();
    renderExperimentPlan(payload);
    recordTraceEvent("paper_experiment_plan", {
      runId: payload.runId,
      plannedSamples: payload.summary?.plannedSamples,
      providerEvalCallsNeeded: payload.summary?.providerEvalCallsNeeded,
    });
    return payload;
  } catch (error) {
    if (!silent) renderExperimentPlanError(error);
    return null;
  } finally {
    if (!silent && els.planRunButton) {
      els.planRunButton.disabled = false;
      els.planRunButton.textContent = "Plan next";
    }
  }
}

async function loadLatestCaseFactory() {
  if (!els.caseFactoryRows) return;
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/case-factory/latest"));
    if (!response.ok) throw new Error("Latest case factory output unavailable");
    const payload = await response.json();
    if (payload?.found === false) return;
    renderCaseFactory(payload);
  } catch {
    // The case factory may not have been generated yet.
  }
}

async function runCaseFactory({ silent = false } = {}) {
  if (!els.caseFactoryRows) return null;
  if (!silent && els.caseFactoryRunButton) {
    els.caseFactoryRunButton.disabled = true;
    els.caseFactoryRunButton.textContent = "Building";
  }
  if (!silent) {
    if (els.caseFactoryRows) els.caseFactoryRows.textContent = "Generating draft cases.";
    if (els.caseFactoryArtifacts) els.caseFactoryArtifacts.textContent = "Case factory running.";
  }
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/case-factory/run"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refreshPlan: false, save: true }),
    });
    if (!response.ok) throw new Error("Case factory failed");
    const payload = await response.json();
    renderCaseFactory(payload);
    recordTraceEvent("paper_case_factory", {
      runId: payload.runId,
      totalDraftArtifacts: payload.summary?.totalDraftArtifacts,
      benchmarkDraftCases: payload.summary?.benchmarkDraftCases,
      speechDraftCases: payload.summary?.speechDraftCases,
    });
    return payload;
  } catch (error) {
    if (!silent) renderCaseFactoryError(error);
    return null;
  } finally {
    if (!silent && els.caseFactoryRunButton) {
      els.caseFactoryRunButton.disabled = false;
      els.caseFactoryRunButton.textContent = "Build cases";
    }
  }
}

async function loadLatestDraftValidation() {
  if (!els.draftValidationRows) return;
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/draft-validation/latest"));
    if (!response.ok) throw new Error("Latest draft validation unavailable");
    const payload = await response.json();
    if (payload?.found === false) return;
    renderDraftValidation(payload);
  } catch {
    // The draft validation gate may not have been generated yet.
  }
}

async function runDraftValidation({ silent = false } = {}) {
  if (!els.draftValidationRows) return null;
  if (!silent && els.draftValidationRunButton) {
    els.draftValidationRunButton.disabled = true;
    els.draftValidationRunButton.textContent = "Validating";
  }
  if (!silent) {
    if (els.draftValidationRows) els.draftValidationRows.textContent = "Scoring generated drafts.";
    if (els.draftValidationArtifacts) els.draftValidationArtifacts.textContent = "Draft validation running.";
  }
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/draft-validation/run"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refreshFactory: false, limit: null, includePayloads: false, save: true }),
    });
    if (!response.ok) throw new Error("Draft validation failed");
    const payload = await response.json();
    renderDraftValidation(payload);
    recordTraceEvent("paper_draft_validation", {
      runId: payload.runId,
      promotionReady: payload.summary?.promotionReady,
      blocked: payload.summary?.blocked,
    });
    return payload;
  } catch (error) {
    if (!silent) renderDraftValidationError(error);
    return null;
  } finally {
    if (!silent && els.draftValidationRunButton) {
      els.draftValidationRunButton.disabled = false;
      els.draftValidationRunButton.textContent = "Validate drafts";
    }
  }
}

async function runPaperReport() {
  if (!els.reportRunButton) return;
  els.reportRunButton.disabled = true;
  els.reportRunButton.textContent = "Generating";
  if (els.reportCoreRows) els.reportCoreRows.textContent = "Generating paper metrics.";
  if (els.reportCostRows) els.reportCostRows.textContent = "Calculating architecture costs.";
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/report/run"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        rerunSuites: Boolean(els.reportRerunInput?.checked),
        save: true,
      }),
    });
    if (!response.ok) throw new Error("Paper report generation failed");
    const payload = await response.json();
    renderPaperReport(payload);
    recordTraceEvent("paper_results_pack", {
      runId: payload.runId,
      readinessScore: payload.summary?.combined?.readinessScore,
      totalCases: payload.summary?.combined?.totalCases,
      costPer1000: payload.summary?.combined?.costPer1000Turns,
      voiceP95Ms: payload.summary?.combined?.voiceP95Ms,
    });
  } catch (error) {
    renderPaperReportError(error);
  } finally {
    els.reportRunButton.disabled = false;
    els.reportRunButton.textContent = "Generate pack";
  }
}

function renderPaperReport(payload) {
  state.latestPaperReport = payload;
  const summary = payload?.summary || {};
  const combined = summary.combined || {};
  const benchmark = summary.benchmark || {};
  const speech = summary.speech || {};
  if (els.reportReadinessMetric) els.reportReadinessMetric.textContent = `${ratioText(combined.readinessScore, 1)} / 100`;
  if (els.reportCasesMetric) els.reportCasesMetric.textContent = String(combined.totalCases ?? "-");
  if (els.reportTaskPassMetric) els.reportTaskPassMetric.textContent = formatPercent(benchmark.passRate);
  if (els.reportSpeechPassMetric) els.reportSpeechPassMetric.textContent = formatPercent(speech.passRate);
  if (els.reportCostMetric) els.reportCostMetric.textContent = formatMoney(combined.costPer1000Turns || 0);
  if (els.reportVoiceMetric) els.reportVoiceMetric.textContent = formatMs(combined.voiceP95Ms);
  if (els.reportRunId) els.reportRunId.textContent = `${payload.runId || "paper-report"} / benchmark ${payload.inputs?.benchmarkRunId || "-"} / speech ${payload.inputs?.speechRunId || "-"}`;
  if (els.reportArtifacts) {
    const artifacts = payload.artifacts || {};
    els.reportArtifacts.textContent = `${artifacts.markdown || "markdown pending"} / ${artifacts.json || "json pending"}`;
  }
  renderReportCoreRows(payload.tables?.coreMetrics || []);
  renderReportCostRows(payload.tables?.costComparison || []);
  if (payload.tables?.audioQualityRows || payload.summary?.audioQuality) {
    renderAudioQuality({
      runId: payload.runId,
      summary: payload.summary?.audioQuality || {},
      retakeQueue: payload.tables?.audioQualityRows || [],
      artifacts: {
        json: payload.artifacts?.audioQualityJson,
        csv: payload.artifacts?.audioQualityCsv,
        retakeQueueJson: payload.artifacts?.retakeQueueJson,
        retakeQueueCsv: payload.artifacts?.retakeQueueCsv,
      },
    });
  }
  if (payload.tables?.audioErrorActionPlan || payload.summary?.audioErrorAnalysis) {
    renderAudioErrorAnalysis({
      runId: payload.runId,
      summary: payload.summary?.audioErrorAnalysis || {},
      rows: payload.tables?.audioErrorRows || [],
      actionPlan: payload.tables?.audioErrorActionPlan || [],
      conditionRisks: payload.tables?.audioErrorConditionRisks || [],
      acceptedCoverageGaps: payload.tables?.audioErrorCoverageGaps || [],
      artifacts: {
        json: payload.artifacts?.audioErrorJson,
        csv: payload.artifacts?.audioErrorCsv,
        actionPlanCsv: payload.artifacts?.audioErrorActionPlanCsv,
      },
    });
  }
  if (payload.tables?.statisticsIntervals) {
    renderStatisticsPack({
      runId: payload.runId,
      summary: payload.summary?.statistics || {},
      metrics: payload.tables.statisticsIntervals,
      artifacts: {
        json: payload.artifacts?.statisticsJson,
        csv: payload.artifacts?.statisticsCsv,
      },
    });
  }
  if (payload.tables?.claimReadiness) {
    renderClaimReadiness({
      runId: payload.runId,
      summary: payload.summary?.claims || {},
      claims: payload.tables.claimReadiness,
      actionPlan: payload.tables.claimActionPlan || [],
      artifacts: {
        json: payload.artifacts?.claimsJson,
        csv: payload.artifacts?.claimsCsv,
      },
    });
  }
  if (payload.tables?.experimentPlan) {
    renderExperimentPlan({
      runId: payload.runId,
      summary: payload.summary?.experimentPlan || {},
      workItems: payload.tables.experimentPlan,
      phases: payload.tables.experimentPhases || [],
      recordingQueue: payload.tables.recordingQueue || [],
      artifacts: {
        json: payload.artifacts?.experimentPlanJson,
        csv: payload.artifacts?.experimentPlanCsv,
      },
    });
  }
  if (payload.tables?.caseFactory) {
    renderCaseFactory({
      runId: payload.runId,
      summary: payload.summary?.caseFactory || {},
      benchmarkCases: (payload.tables.caseFactoryRows || []).filter((row) => row.type),
      speechCases: (payload.tables.caseFactoryRows || []).filter((row) => row.referenceText && row.transcriptText),
      audioRecordingPrompts: (payload.tables.caseFactoryRows || []).filter((row) => row.recordingInstruction),
      artifacts: {
        json: payload.artifacts?.caseFactoryJson,
        csv: payload.artifacts?.caseFactoryCsv,
        benchmarkDrafts: payload.artifacts?.benchmarkDrafts,
        speechDrafts: payload.artifacts?.speechDrafts,
        audioPromptQueue: payload.artifacts?.audioPromptQueue,
      },
    });
  }
  if (payload.tables?.draftValidation) {
    renderDraftValidation({
      runId: payload.runId,
      summary: payload.summary?.draftValidation || {},
      rows: payload.tables.draftValidationRows || [],
      artifacts: {
        json: payload.artifacts?.draftValidationJson,
        csv: payload.artifacts?.draftValidationCsv,
        promotionManifestJson: payload.artifacts?.promotionManifestJson,
        promotionManifestCsv: payload.artifacts?.promotionManifestCsv,
      },
    });
  }
  if (payload.tables?.suitePromotion) {
    renderSuitePromotion({
      runId: payload.runId,
      summary: payload.summary?.suitePromotion || {},
      rows: payload.tables.suitePromotion || [],
      skipped: payload.tables.suitePromotionSkipped || [],
      artifacts: {
        json: payload.artifacts?.suitePromotionJson,
        csv: payload.artifacts?.suitePromotionCsv,
        promotedAudioQueue: payload.artifacts?.promotedAudioQueue,
      },
    });
  }
}

function renderReportCoreRows(rows) {
  if (!els.reportCoreRows) return;
  if (!rows.length) {
    els.reportCoreRows.textContent = "No core metrics.";
    return;
  }
  els.reportCoreRows.replaceChildren(
    ...rows.map((row) => {
      const item = document.createElement("article");
      item.className = "benchmark-case-row pass";
      const title = document.createElement("strong");
      title.textContent = `${row.metric}: ${row.display}`;
      const meta = document.createElement("span");
      meta.textContent = `${row.source} / ${row.paperUse}`;
      item.append(title, meta);
      return item;
    }),
  );
}

function renderReportCostRows(rows) {
  if (!els.reportCostRows) return;
  if (!rows.length) {
    els.reportCostRows.textContent = "No cost rows.";
    return;
  }
  els.reportCostRows.replaceChildren(
    ...rows.map((row, index) => {
      const item = document.createElement("article");
      const savings = Number(row.savingsVsComposed);
      const isWorse = Number.isFinite(savings) && savings < 0;
      item.className = `benchmark-case-row ${index === 0 ? "pass" : isWorse ? "review" : "pass"}`;
      const title = document.createElement("strong");
      title.textContent = `${row.architecture}: ${row.display}`;
      const meta = document.createElement("span");
      meta.textContent = `${row.mode} / ${savingsPhrase(row.savingsVsComposed)}`;
      item.append(title, meta);
      return item;
    }),
  );
}

function renderPaperReportError(error) {
  if (els.reportReadinessMetric) els.reportReadinessMetric.textContent = "failed";
  if (els.reportCoreRows) {
    const empty = document.createElement("div");
    empty.className = "trace-empty";
    empty.textContent = readableError(error);
    els.reportCoreRows.replaceChildren(empty);
  }
  if (els.reportCostRows) els.reportCostRows.textContent = "Report generation failed.";
}

function renderStatisticsPack(payload) {
  state.latestStatisticsPack = payload;
  const summary = payload?.summary || {};
  const widest = summary.widestCiMetric || {};
  if (els.statsMetricsMetric) els.statsMetricsMetric.textContent = `${summary.populatedMetricCount ?? 0} / ${summary.metricCount ?? 0}`;
  if (els.statsCoverageMetric) els.statsCoverageMetric.textContent = formatPercent(summary.coverageRate);
  if (els.statsWidestMetric) els.statsWidestMetric.textContent = widest.label || widest.id || "-";
  if (els.statsCiMetric) els.statsCiMetric.textContent = widest.displayCi || "-";
  if (els.statsArtifacts) {
    const artifacts = payload.artifacts || {};
    els.statsArtifacts.textContent = `${artifacts.json || "statistics json pending"} / ${artifacts.csv || "statistics csv pending"}`;
  }
  renderStatisticsRows(payload.metrics || []);
}

function renderStatisticsRows(rows) {
  if (!els.statsRows) return;
  const visibleRows = rows.slice(0, 14);
  if (!visibleRows.length) {
    els.statsRows.textContent = "No statistics rows.";
    return;
  }
  els.statsRows.replaceChildren(
    ...visibleRows.map((row) => {
      const item = document.createElement("article");
      item.className = `benchmark-case-row ${row.status === "ok" ? "pass" : "review"}`;
      const title = document.createElement("strong");
      title.textContent = `${row.label || row.id}: ${row.displayObserved || "-"}`;
      const meta = document.createElement("span");
      meta.textContent = `${row.displayCi || "-"} / n=${row.n ?? 0} / ${row.method || "-"}`;
      const note = document.createElement("span");
      note.textContent = `${row.source || "-"} / ${row.paperUse || "-"}`;
      item.append(title, meta, note);
      return item;
    }),
  );
}

function renderStatisticsError(error) {
  if (els.statsMetricsMetric) els.statsMetricsMetric.textContent = "failed";
  if (els.statsRows) {
    const empty = document.createElement("div");
    empty.className = "trace-empty";
    empty.textContent = readableError(error);
    els.statsRows.replaceChildren(empty);
  }
  if (els.statsArtifacts) els.statsArtifacts.textContent = "Statistics failed.";
}

function renderClaimReadiness(payload) {
  state.latestClaimReadiness = payload;
  const summary = payload?.summary || {};
  const ready = summary.publishable ?? 0;
  const total = summary.totalClaims ?? 0;
  if (els.claimReadyMetric) els.claimReadyMetric.textContent = `${ready} / ${total}`;
  if (els.claimDataMetric) els.claimDataMetric.textContent = String(summary.needsMoreData ?? 0);
  if (els.claimWorkMetric) els.claimWorkMetric.textContent = `${summary.needsSystemWork ?? 0} work / ${summary.missingEvidence ?? 0} missing`;
  if (els.claimActionMetric) els.claimActionMetric.textContent = summary.topAction || "-";
  if (els.claimsArtifacts) {
    const artifacts = payload.artifacts || {};
    els.claimsArtifacts.textContent = `${artifacts.json || "claims json pending"} / ${artifacts.csv || "claims csv pending"}`;
  }
  renderClaimRows(payload.claims || []);
}

function renderClaimRows(rows) {
  if (!els.claimsRows) return;
  if (!rows.length) {
    els.claimsRows.textContent = "No claim readiness rows.";
    return;
  }
  els.claimsRows.replaceChildren(
    ...rows.slice(0, 14).map((row) => {
      const item = document.createElement("article");
      const status = row.status || "review";
      item.className = `benchmark-case-row ${status === "publishable" ? "pass" : status === "needs_system_work" || status === "missing_evidence" ? "fail" : "review"}`;
      const title = document.createElement("strong");
      title.textContent = `${row.section || "Claim"}: ${row.status || "-"} - ${row.displayObserved || "-"}`;
      const claim = document.createElement("span");
      claim.textContent = row.claim || row.id || "-";
      const meta = document.createElement("span");
      meta.textContent = `${row.displayCi || "-"} / n=${row.n ?? 0} / add ${row.additionalSamples ?? 0}`;
      const action = document.createElement("span");
      action.textContent = row.nextAction || "-";
      item.append(title, claim, meta, action);
      return item;
    }),
  );
}

function renderClaimReadinessError(error) {
  if (els.claimReadyMetric) els.claimReadyMetric.textContent = "failed";
  if (els.claimsRows) {
    const empty = document.createElement("div");
    empty.className = "trace-empty";
    empty.textContent = readableError(error);
    els.claimsRows.replaceChildren(empty);
  }
  if (els.claimsArtifacts) els.claimsArtifacts.textContent = "Claims failed.";
}

function renderExperimentPlan(payload) {
  state.latestExperimentPlan = payload;
  const summary = payload?.summary || {};
  const textTotal = Number(summary.benchmarkCasesToAdd || 0) + Number(summary.speechProxyCasesToAdd || 0);
  const audioTotal = Number(summary.realAudioRecordingsToAdd || 0) + Number(summary.stressPairsToEvaluate || 0);
  if (els.planSamplesMetric) els.planSamplesMetric.textContent = String(summary.plannedSamples ?? "-");
  if (els.planTextMetric) els.planTextMetric.textContent = `${textTotal} cases`;
  if (els.planAudioMetric) els.planAudioMetric.textContent = `${audioTotal} items`;
  if (els.planProviderMetric) els.planProviderMetric.textContent = String(summary.providerEvalCallsNeeded ?? "-");
  if (els.planArtifacts) {
    const artifacts = payload.artifacts || {};
    els.planArtifacts.textContent = `${artifacts.json || "plan json pending"} / ${artifacts.csv || "plan csv pending"}`;
  }
  renderExperimentPlanRows(payload.workItems || []);
}

function renderExperimentPlanRows(rows) {
  if (!els.planRows) return;
  if (!rows.length) {
    els.planRows.textContent = "No experiment plan rows.";
    return;
  }
  els.planRows.replaceChildren(
    ...rows.slice(0, 12).map((row) => {
      const item = document.createElement("article");
      item.className = `benchmark-case-row ${Number(row.priority || 0) >= 90 ? "fail" : Number(row.addCount || 0) > 0 ? "review" : "pass"}`;
      const title = document.createElement("strong");
      title.textContent = `${row.lane || "plan"}: add ${row.addCount ?? 0}`;
      const action = document.createElement("span");
      action.textContent = row.action || row.id || "-";
      const reason = document.createElement("span");
      reason.textContent = row.reason || "-";
      const use = document.createElement("span");
      use.textContent = row.paperUse || "-";
      item.append(title, action, reason, use);
      return item;
    }),
  );
}

function renderExperimentPlanError(error) {
  if (els.planSamplesMetric) els.planSamplesMetric.textContent = "failed";
  if (els.planRows) {
    const empty = document.createElement("div");
    empty.className = "trace-empty";
    empty.textContent = readableError(error);
    els.planRows.replaceChildren(empty);
  }
  if (els.planArtifacts) els.planArtifacts.textContent = "Plan failed.";
}

function renderCaseFactory(payload) {
  state.latestCaseFactory = payload;
  const summary = payload?.summary || {};
  if (els.caseFactoryTotalMetric) els.caseFactoryTotalMetric.textContent = String(summary.totalDraftArtifacts ?? "-");
  if (els.caseFactoryBenchmarkMetric) els.caseFactoryBenchmarkMetric.textContent = String(summary.benchmarkDraftCases ?? "-");
  if (els.caseFactorySpeechMetric) els.caseFactorySpeechMetric.textContent = String(summary.speechDraftCases ?? "-");
  if (els.caseFactoryAudioMetric) els.caseFactoryAudioMetric.textContent = String(summary.audioRecordingPrompts ?? "-");
  if (els.caseFactoryArtifacts) {
    const artifacts = payload.artifacts || {};
    els.caseFactoryArtifacts.textContent = `${artifacts.benchmarkDrafts || artifacts.json || "benchmark drafts pending"} / ${artifacts.speechDrafts || "speech drafts pending"} / ${artifacts.audioPromptQueue || "audio prompts pending"}`;
  }
  renderCaseFactoryRows([
    ...(payload.benchmarkCases || []).slice(0, 5).map((row) => ({ ...row, artifactType: "benchmark" })),
    ...(payload.speechCases || []).slice(0, 5).map((row) => ({ ...row, artifactType: "speech" })),
    ...(payload.audioRecordingPrompts || []).slice(0, 5).map((row) => ({ ...row, artifactType: "audio" })),
  ]);
}

function renderCaseFactoryRows(rows) {
  if (!els.caseFactoryRows) return;
  if (!rows.length) {
    els.caseFactoryRows.textContent = "No draft cases.";
    return;
  }
  els.caseFactoryRows.replaceChildren(
    ...rows.map((row) => {
      const item = document.createElement("article");
      item.className = "benchmark-case-row review";
      const title = document.createElement("strong");
      title.textContent = `${row.artifactType || "draft"}: ${row.id || row.templateId || "-"}`;
      const text = document.createElement("span");
      text.textContent = row.query || row.referenceText || row.recordingInstruction || "-";
      const meta = document.createElement("span");
      meta.textContent = `${row.group || row.route || "-"} / ${row.condition?.accent || row.recommendedStratum || row.condition || "draft"}`;
      item.append(title, text, meta);
      return item;
    }),
  );
}

function renderCaseFactoryError(error) {
  if (els.caseFactoryTotalMetric) els.caseFactoryTotalMetric.textContent = "failed";
  if (els.caseFactoryRows) {
    const empty = document.createElement("div");
    empty.className = "trace-empty";
    empty.textContent = readableError(error);
    els.caseFactoryRows.replaceChildren(empty);
  }
  if (els.caseFactoryArtifacts) els.caseFactoryArtifacts.textContent = "Case factory failed.";
}

function renderDraftValidation(payload) {
  state.latestDraftValidation = payload;
  const summary = payload?.summary || {};
  if (els.draftReadyMetric) els.draftReadyMetric.textContent = `${summary.promotionReady ?? 0} / ${summary.totalDraftArtifacts ?? 0}`;
  if (els.draftBlockedMetric) els.draftBlockedMetric.textContent = String(summary.blocked ?? 0);
  if (els.draftBenchmarkMetric) els.draftBenchmarkMetric.textContent = `${summary.benchmarkPromotionReady ?? 0} / ${summary.benchmarkDrafts ?? 0}`;
  if (els.draftSpeechMetric) els.draftSpeechMetric.textContent = `${summary.speechPromotionReady ?? 0} / ${summary.speechDrafts ?? 0}`;
  if (els.draftValidationArtifacts) {
    const artifacts = payload.artifacts || {};
    els.draftValidationArtifacts.textContent = `${artifacts.json || "validation json pending"} / ${artifacts.promotionManifestJson || "promotion manifest pending"}`;
  }
  renderDraftValidationRows(payload.rows || []);
}

function renderDraftValidationRows(rows) {
  if (!els.draftValidationRows) return;
  if (!rows.length) {
    els.draftValidationRows.textContent = "No draft validation rows.";
    return;
  }
  els.draftValidationRows.replaceChildren(
    ...rows.slice(0, 16).map((row) => {
      const item = document.createElement("article");
      item.className = `benchmark-case-row ${row.promotionReady ? "pass" : "fail"}`;
      const title = document.createElement("strong");
      title.textContent = `${row.artifactType || "draft"}: ${row.status || "-"} - ${row.id || "-"}`;
      const text = document.createElement("span");
      text.textContent = row.query || row.referenceText || row.recommendedStratum || "-";
      const meta = document.createElement("span");
      meta.textContent = `${row.group || "-"} / failures ${row.failureCount ?? 0}`;
      const failures = document.createElement("span");
      failures.textContent = Array.isArray(row.failures) && row.failures.length ? row.failures.join(" | ") : "promotion ready";
      item.append(title, text, meta, failures);
      return item;
    }),
  );
}

function renderDraftValidationError(error) {
  if (els.draftReadyMetric) els.draftReadyMetric.textContent = "failed";
  if (els.draftValidationRows) {
    const empty = document.createElement("div");
    empty.className = "trace-empty";
    empty.textContent = readableError(error);
    els.draftValidationRows.replaceChildren(empty);
  }
  if (els.draftValidationArtifacts) els.draftValidationArtifacts.textContent = "Draft validation failed.";
}

async function loadLatestSuitePromotion() {
  if (!els.promotionRows) return;
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/promotion/latest"));
    if (!response.ok) throw new Error("Latest promotion preview unavailable");
    const payload = await response.json();
    if (payload?.found === false) return;
    renderSuitePromotion(payload);
  } catch {
    // The promotion preview may not exist yet.
  }
}

async function runSuitePromotion({ silent = false } = {}) {
  if (!els.promotionRows) return null;
  const writeSuiteFiles = Boolean(els.promotionWriteInput?.checked);
  if (!silent && els.promotionRunButton) {
    els.promotionRunButton.disabled = true;
    els.promotionRunButton.textContent = writeSuiteFiles ? "Promoting" : "Previewing";
  }
  if (!silent) {
    if (els.promotionRows) els.promotionRows.textContent = writeSuiteFiles ? "Writing validated cases to suite files." : "Previewing validated case promotion.";
    if (els.promotionArtifacts) els.promotionArtifacts.textContent = "Promotion gate running.";
  }
  try {
    const response = await fetch(previewBackendUrl("/api/evaluation/promotion/run"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dryRun: !writeSuiteFiles,
        replaceFactoryCases: true,
        includeBenchmark: true,
        includeSpeech: true,
        includeAudioQueue: true,
        refreshValidation: false,
        save: true,
      }),
    });
    if (!response.ok) throw new Error("Suite promotion failed");
    const payload = await response.json();
    renderSuitePromotion(payload);
    recordTraceEvent("suite_promotion", {
      runId: payload.runId,
      dryRun: payload.summary?.dryRun,
      totalAdded: payload.summary?.totalAdded,
      totalSkipped: payload.summary?.totalSkipped,
      wroteFiles: payload.summary?.wroteFiles,
    });
    return payload;
  } catch (error) {
    if (!silent) renderSuitePromotionError(error);
    return null;
  } finally {
    if (!silent && els.promotionRunButton) {
      els.promotionRunButton.disabled = false;
      els.promotionRunButton.textContent = writeSuiteFiles ? "Promote suite" : "Preview promote";
    }
  }
}

function renderSuitePromotion(payload) {
  state.latestSuitePromotion = payload;
  const summary = payload?.summary || {};
  const dryRun = Boolean(summary.dryRun);
  if (els.promotionModeMetric) els.promotionModeMetric.textContent = dryRun ? "dry run" : "wrote";
  if (els.promotionAddedMetric) els.promotionAddedMetric.textContent = String(summary.totalAdded ?? "-");
  if (els.promotionSkippedMetric) els.promotionSkippedMetric.textContent = String(summary.totalSkipped ?? 0);
  if (els.promotionWrittenMetric) els.promotionWrittenMetric.textContent = String(summary.wroteFiles ?? 0);
  if (els.promotionArtifacts) {
    const artifacts = payload.artifacts || {};
    els.promotionArtifacts.textContent = `${artifacts.json || "promotion json pending"} / ${artifacts.csv || "promotion csv pending"}`;
  }
  renderSuitePromotionRows(payload.rows || [], payload.skipped || []);
}

function renderSuitePromotionRows(rows, skippedRows = []) {
  if (!els.promotionRows) return;
  if (!rows.length) {
    els.promotionRows.textContent = "No promotion preview rows.";
    return;
  }
  const targetRows = rows.map((row) => {
    const item = document.createElement("article");
    item.className = `benchmark-case-row ${Number(row.skippedCount || 0) ? "review" : "pass"}`;
    const title = document.createElement("strong");
    title.textContent = `${row.target || "target"}: add ${row.addedCount ?? 0} / ${row.candidateCount ?? 0}`;
    const meta = document.createElement("span");
    meta.textContent = `${row.dryRun ? "dry run" : "wrote"} / replace ${row.replacedFactoryCount ?? 0} / skipped ${row.skippedCount ?? 0}`;
    const path = document.createElement("span");
    path.textContent = row.backupPath ? `${row.targetPath} / backup ${row.backupPath}` : row.targetPath || "-";
    item.append(title, meta, path);
    return item;
  });
  const skipped = skippedRows.slice(0, 6).map((row) => {
    const item = document.createElement("article");
    item.className = "benchmark-case-row review";
    const title = document.createElement("strong");
    title.textContent = `Skipped: ${row.id || "-"}`;
    const meta = document.createElement("span");
    meta.textContent = `${row.target || "-"} / ${row.reason || "-"}`;
    item.append(title, meta);
    return item;
  });
  els.promotionRows.replaceChildren(...targetRows, ...skipped);
}

function renderSuitePromotionError(error) {
  if (els.promotionModeMetric) els.promotionModeMetric.textContent = "failed";
  if (els.promotionRows) {
    const empty = document.createElement("div");
    empty.className = "trace-empty";
    empty.textContent = readableError(error);
    els.promotionRows.replaceChildren(empty);
  }
  if (els.promotionArtifacts) els.promotionArtifacts.textContent = "Promotion failed.";
}

function savingsPhrase(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "baseline unavailable";
  if (number === 0) return "composed stack baseline";
  if (number > 0) return `${formatPercent(number)} lower than this baseline`;
  return `${formatPercent(Math.abs(number))} higher than this baseline`;
}
async function loadCatalog() {
  try {
    const response = await fetch(previewBackendUrl("/api/catalog"));
    const items = await response.json();
    const catalogItems = items.map((item) => item.synonyms?.[0] || item.name);
    const names = [...new Set([...sampleItems, ...catalogItems])];
    renderCatalog(names.slice(0, 10));
  } catch {
    renderCatalog(sampleItems);
  }
}

async function loadCatalogSummary() {
  try {
    const response = await fetch(previewBackendUrl("/api/catalog/summary"));
    if (!response.ok) throw new Error("Catalog summary unavailable");
    renderCatalogSummary(await response.json());
  } catch {
    renderCatalogSummary({ totalProducts: "-", departments: {}, availability: {} });
  }
}

function renderCatalog(items) {
  els.catalogStrip.replaceChildren(
    ...items.map((name) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = name;
      button.addEventListener("click", async () => {
        const prompt = `Where is ${name}, and is it in stock?`;
        const previewTrace = !state.callActive;
        if (previewTrace) {
          beginTrace("local_preview");
          state.userText = prompt;
          state.answer = "";
          recordTraceEvent("transcript", { role: "user", transcriptType: "final", text: prompt, turn: 1 });
        }
        els.transcriptText.textContent = prompt;
        els.answerText.textContent = "Local preview result. Start voice session for spoken output.";
        els.toolItem.textContent = `Looking up ${name}`;
        els.toolAisle.textContent = "...";
        els.toolStock.textContent = "...";
        els.toolMatch.textContent = "...";
        setStage("llm", "Local inventory preview");
        try {
          const response = await fetch(previewBackendUrl("/api/inventory/lookup"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: name }),
          });
          if (!response.ok) throw new Error("Preview lookup failed");
          recordTraceEvent("tool_call", { name: "lookup_inventory", query: name, arguments: { query: name } });
          const payload = await response.json();
          recordTraceEvent("tool_result", payload);
          state.answer = payload.speechAnswer || "";
          renderTool(payload);
          setStage("idle", "Preview complete");
          if (previewTrace) {
            recordTraceEvent("transcript", {
              role: "assistant",
              transcriptType: "final",
              text: state.answer || "Local preview result.",
              turn: 1,
            });
            state.turnCount = 1;
            state.metrics = {
              ...state.metrics,
              turn: 1,
              totalMs: 0,
              vadMs: 0,
              sttMs: 0,
              voiceMs: 0,
            };
            recordTraceEvent("turn_complete", { turn: 1, userText: state.userText, answer: state.answer, metrics: state.metrics });
            await refreshLedger();
            await saveCurrentTrace("preview");
          }
        } catch {
          recordTraceEvent("error", { stage: "local_preview", message: "Local preview could not reach the local service." });
          els.answerText.textContent = "Local preview could not reach the local service.";
          setStage("idle", "Preview failed");
        }
        if (state.callActive) {
          state.vapi?.send({
            type: "add-message",
            message: { role: "user", content: prompt },
            triggerResponseEnabled: true,
          });
        }
      });
      return button;
    }),
  );
}

function seedLedger() {
  renderLedger({
    rows: [
      { label: "Composed voice agent", mode: "live estimate", cost: 0, per1000: 0, latencyMs: 0 },
      { label: "Native realtime baseline", mode: "published pricing baseline", cost: 0, per1000: 0, latencyMs: 950 },
      { label: "Native audio baseline", mode: "published pricing baseline", cost: 0, per1000: 0, latencyMs: 900 },
    ],
    assumptions: {},
  });
}

function formatMs(value) {
  return Number.isFinite(Number(value)) ? `${Math.round(Number(value))} ms` : "-";
}

function formatMoney(value) {
  if (!Number.isFinite(value)) return "-";
  if (value > 0 && value < 0.0001) return "<$0.0001";
  return `$${value.toFixed(4)}`;
}

function readableError(error) {
  if (typeof error === "string") return error;
  return error?.message || error?.error?.message || "The voice session could not start.";
}


restoreConfig();
loadPublicVoiceConfig();
els.openAgentButtons.forEach((button) => {
  button.addEventListener("click", () => openSimpleAgent({ autoStart: true }));
});
els.openLabButtons.forEach((button) => {
  button.addEventListener("click", (event) => {
    event.preventDefault();
    const url = new URL(window.location.href);
    url.searchParams.set("lab", "1");
    window.history.pushState({}, "", url);
    openAgentWorkspace(button.dataset.agentTarget || "voice");
  });
});
els.simpleCallButton?.addEventListener("click", () => toggleSimpleCall());
els.simpleEndButton?.addEventListener("click", async () => {
  if (state.callActive || state.starting) {
    await endCall("manual");
    return;
  }
  closeSimpleAgent();
});
els.simpleCloseButton?.addEventListener("click", () => closeSimpleAgent());
els.feedbackButtons.forEach((button) => {
  button.addEventListener("click", () => selectFeedbackScore(button.dataset.feedbackScore));
});
els.feedbackSubmitButton?.addEventListener("click", () => submitFeedback());
els.agentCloseButton?.addEventListener("click", () => closeAgentWorkspace());
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && document.body.classList.contains("voice-call-open")) closeSimpleAgent();
  if (event.key === "Escape" && document.body.classList.contains("agent-open")) closeAgentWorkspace();
});
els.callButton.addEventListener("click", () => toggleCall());
els.callButtonTop.addEventListener("click", () => toggleCall());
els.endpointingButtons.forEach((button) => {
  button.addEventListener("click", () => setEndpointingMode(button.dataset.endpointingMode || "balanced"));
});
if (els.benchmarkRunButton) {
  els.benchmarkRunButton.addEventListener("click", () => runBenchmarkSuite());
}
if (els.speechRunButton) {
  els.speechRunButton.addEventListener("click", () => runSpeechSuite());
}
if (els.audioRunButton) {
  els.audioRunButton.addEventListener("click", () => runAudioSuite());
}
if (els.audioRecordButton) {
  els.audioRecordButton.addEventListener("click", () => toggleAudioRecording());
}
if (els.audioSaveButton) {
  els.audioSaveButton.addEventListener("click", () => saveAudioRecording());
}
if (els.audioManifestButton) {
  els.audioManifestButton.addEventListener("click", () => buildAudioManifest());
}
if (els.audioQualityButton) {
  els.audioQualityButton.addEventListener("click", () => runAudioQuality());
}
if (els.audioAcceptedButton) {
  els.audioAcceptedButton.addEventListener("click", () => runAudioAcceptedSet());
}
if (els.audioErrorButton) {
  els.audioErrorButton.addEventListener("click", () => runAudioErrorAnalysis());
}
if (els.audioAccentSweepButton) {
  els.audioAccentSweepButton.addEventListener("click", () => runAudioAccentSweep());
}
if (els.audioStressButton) {
  els.audioStressButton.addEventListener("click", () => generateAudioStressVariants());
}
if (els.audioRobustnessButton) {
  els.audioRobustnessButton.addEventListener("click", () => runAudioRobustness());
}
if (els.audioCaseSelect) {
  els.audioCaseSelect.addEventListener("change", () => syncAudioReference());
}
if (els.reportRunButton) {
  els.reportRunButton.addEventListener("click", () => runPaperReport());
}
if (els.statsRunButton) {
  els.statsRunButton.addEventListener("click", () => runStatisticsPack());
}
if (els.claimsRunButton) {
  els.claimsRunButton.addEventListener("click", () => runClaimReadiness());
}
if (els.planRunButton) {
  els.planRunButton.addEventListener("click", () => runExperimentPlan());
}
if (els.caseFactoryRunButton) {
  els.caseFactoryRunButton.addEventListener("click", () => runCaseFactory());
}
if (els.draftValidationRunButton) {
  els.draftValidationRunButton.addEventListener("click", () => runDraftValidation());
}
if (els.promotionRunButton) {
  els.promotionRunButton.addEventListener("click", () => runSuitePromotion());
}
if (els.promotionWriteInput && els.promotionRunButton) {
  els.promotionWriteInput.addEventListener("change", () => {
    els.promotionRunButton.textContent = els.promotionWriteInput.checked ? "Promote suite" : "Preview promote";
  });
}
if (els.ragRunButton) {
  els.ragRunButton.addEventListener("click", () => runRagLab());
}
if (els.ragFaithfulnessButton) {
  els.ragFaithfulnessButton.addEventListener("click", () => runFaithfulnessGrade());
}

if (new URLSearchParams(window.location.search).get("lab") === "1") {
  window.setTimeout(() => openAgentWorkspace("voice"), 100);
}
if (els.ragQueryInput) {
  els.ragQueryInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") runRagLab();
  });
}
els.toolServerUrl.addEventListener("change", () => {
  saveConfig();
  loadBenchmarkCases();
  loadLatestBenchmark();
  loadSpeechCases();
  loadLatestSpeechEval();
  loadAudioCases();
  loadLatestAudioEval();
  loadLatestAudioManifest();
  loadLatestAudioQuality();
  loadLatestAudioAcceptedSet();
  loadLatestAudioErrorAnalysis();
  loadLatestAudioAccentSweep();
  loadLatestAudioRobustness();
  loadLatestPaperReport();
  loadLatestStatisticsPack();
  loadLatestClaimReadiness();
  loadLatestExperimentPlan();
  loadLatestCaseFactory();
  loadLatestDraftValidation();
  loadLatestSuitePromotion();
  loadCatalog();
  loadCatalogSummary();
  loadFeedbackSummary();
});
loadBenchmarkCases();
loadLatestBenchmark();
loadSpeechCases();
loadLatestSpeechEval();
  loadAudioCases();
  loadLatestAudioEval();
  loadLatestAudioManifest();
  loadLatestAudioQuality();
  loadLatestAudioAcceptedSet();
  loadLatestAudioErrorAnalysis();
  loadLatestAudioAccentSweep();
  loadLatestAudioRobustness();
  loadLatestPaperReport();
  loadLatestStatisticsPack();
  loadLatestClaimReadiness();
  loadLatestExperimentPlan();
  loadLatestCaseFactory();
  loadLatestDraftValidation();
  loadLatestSuitePromotion();
  loadCatalog();
loadCatalogSummary();
loadFeedbackSummary();
seedLedger();
loadTraces();
updateTracePanel();































