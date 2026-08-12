import { useState } from "react";
import type { AudioAnalysisResult } from "../types/audio";
import { uploadAudio } from "../services/api";

export function AudioUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AudioAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const selected = e.target.files?.[0] ?? null;
    setFile(selected);
    setResult(null);
    setError(null);
  }

  async function handleUpload() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await uploadAudio(file);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ padding: "1rem", border: "1px solid #ccc", borderRadius: "8px", maxWidth: "480px" }}>
      <h2>Upload Audio</h2>
      <input type="file" accept=".wav,.mp3,.flac" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={!file || loading} style={{ marginLeft: "1rem" }}>
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {error && <p style={{ color: "red" }}>Error: {error}</p>}

      {result && (
        <div style={{ marginTop: "1rem" }}>
          <p><strong>File:</strong> {result.filename}</p>
          <p><strong>Duration:</strong> {result.duration_seconds.toFixed(2)}s</p>
          <p><strong>Spectral centroid (mean):</strong> {result.rir_features.spectral_centroid_mean.toFixed(2)}</p>
          <p><strong>Spectral flatness (mean):</strong> {result.rir_features.spectral_flatness_mean.toExponential(3)}</p>
          <p><strong>RMS energy (std):</strong> {result.rir_features.rms_energy_std.toFixed(4)}</p>
        </div>
      )}
    </div>
  );
}