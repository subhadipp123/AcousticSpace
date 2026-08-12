import type { AudioAnalysisResult } from "../types/audio";

const API_BASE_URL = "http://127.0.0.1:8000";

export async function uploadAudio(file: File): Promise<AudioAnalysisResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => null);
    throw new Error(errorBody?.detail || `Upload failed with status ${response.status}`);
  }

  return response.json();
}