export interface RirFeatures {
  spectral_centroid_mean: number;
  spectral_centroid_std: number;
  spectral_flatness_mean: number;
  rms_energy_std: number;
}

export interface AudioAnalysisResult {
  filename: string;
  duration_seconds: number;
  rir_features: RirFeatures;
}