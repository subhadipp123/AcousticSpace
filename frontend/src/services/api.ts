import type {
  AudioAnalysisResult,
  AnalysisHistoryItem,
} from "../types/audio";


export const API_BASE_URL =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:8000";


export async function uploadAudio(
  file: File,
): Promise<AudioAnalysisResult> {
  const formData = new FormData();

  formData.append(
    "file",
    file,
  );

  const response = await fetch(
    `${API_BASE_URL}/upload`,
    {
      method: "POST",
      body: formData,
    },
  );

  if (!response.ok) {
    let message =
      `Upload failed (${response.status})`;

    try {
      const errorData =
        await response.json();

      if (errorData.detail) {
        message = errorData.detail;
      }
    } catch {
      // Keep default message.
    }

    throw new Error(message);
  }

  return response.json() as Promise<AudioAnalysisResult>;
}


export async function fetchHistory(): Promise<
  AnalysisHistoryItem[]
> {
  const response = await fetch(
    `${API_BASE_URL}/history`,
  );

  if (!response.ok) {
    throw new Error(
      `Failed to load history (${response.status})`,
    );
  }

  const data = (await response.json()) as {
    items: AnalysisHistoryItem[];
  };

  return data.items;
}