export interface RIRFeatures {
  spectral_centroid_mean: number;
  spectral_centroid_std: number;
  spectral_flatness_mean: number;
  rms_energy_std: number;
}

export interface ModelPrediction {
  label: "bonafide" | "spoof";
  class_id: number;
  confidence: number;
  bonafide_probability: number;
  spoof_probability: number;
  model: "CNN" | "AST";
}

export interface AudioSegment {
  start_seconds: number;
  end_seconds: number;
  prediction: "bonafide" | "spoof";
  confidence: number;
  bonafide_probability: number;
  spoof_probability: number;
  suspicious: boolean;
}

export interface AudioAnalysisResult {
  filename: string;
  duration_seconds: number;

  audio_url: string;

  rir_features: RIRFeatures;

  cnn: ModelPrediction;
  ast: ModelPrediction;
  primary_prediction: ModelPrediction;

  segments: AudioSegment[];

  spectrogram_path: string;
}

export interface AnalysisHistoryItem {
  id: string;
  filename: string;
  timestamp: string;
  prediction: "bonafide" | "spoof";
  confidence: number;
  model: "CNN" | "AST";
}