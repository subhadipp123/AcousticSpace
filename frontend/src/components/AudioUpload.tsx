import {
  useEffect,
  useState,
} from "react";

import type {
  AudioAnalysisResult,
  AnalysisHistoryItem,
} from "../types/audio";

import {
  fetchHistory,
  uploadAudio,
} from "../services/api";

import { API_BASE_URL } from "../services/api";

import { Waveform } from "./Waveform";


export function AudioUpload() {
  const [file, setFile] =
    useState<File | null>(null);

  const [result, setResult] =
    useState<AudioAnalysisResult | null>(
      null,
    );

  const [error, setError] =
    useState<string | null>(null);

  const [loading, setLoading] =
    useState(false);

  const [history, setHistory] =
    useState<AnalysisHistoryItem[]>(
      [],
    );


  useEffect(() => {
    fetchHistory()
      .then((items) => {
        setHistory(items);
      })
      .catch(() => {
        // Keep dashboard usable.
      });
  }, []);


  function handleFileChange(
    e: React.ChangeEvent<HTMLInputElement>,
  ) {
    const selected =
      e.target.files?.[0] ?? null;

    setFile(selected);
    setResult(null);
    setError(null);
  }


  async function handleUpload() {
    if (!file) {
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data =
        await uploadAudio(file);

      setResult(data);

      const updatedHistory =
        await fetchHistory();

      setHistory(updatedHistory);

    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Upload failed",
      );
    } finally {
      setLoading(false);
    }
  }


  return (
    <div
      style={{
        padding: "1.5rem",
        border: "1px solid #ccc",
        borderRadius: "10px",
        maxWidth: "700px",
        width: "100%",
        boxSizing: "border-box",
      }}
    >
      <h2>Upload Audio</h2>

      <input
        type="file"
        accept=".wav,.mp3,.flac"
        onChange={handleFileChange}
      />

      <button
        onClick={handleUpload}
        disabled={!file || loading}
        style={{
          marginLeft: "1rem",
          padding: "0.5rem 1rem",
          cursor:
            !file || loading
              ? "not-allowed"
              : "pointer",
        }}
      >
        {loading
          ? "Analyzing..."
          : "Analyze"}
      </button>

      {error && (
        <p
          style={{
            color: "red",
            marginTop: "1rem",
          }}
        >
          Error: {error}
        </p>
      )}

      {result && (
        <div
          style={{
            marginTop: "1.5rem",
          }}
        >
          <h3>
            Analysis Result
          </h3>

          <p>
            <strong>File:</strong>{" "}
            {result.filename}
          </p>

          <p>
            <strong>Duration:</strong>{" "}
            {result.duration_seconds.toFixed(
              2,
            )}
            s
          </p>

          <hr />

          <Waveform
            audioUrl={`${API_BASE_URL}${result.audio_url}`}
            segments={result.segments}
          />

          <hr />

          <h4>
            Primary Prediction — AST
          </h4>

          <p>
            <strong>
              Prediction:
            </strong>{" "}
            {result.primary_prediction.label.toUpperCase()}
          </p>

          <p>
            <strong>
              Confidence:
            </strong>{" "}
            {(
              result.primary_prediction
                .confidence * 100
            ).toFixed(2)}
            %
          </p>

          <p>
            <strong>
              Bonafide probability:
            </strong>{" "}
            {(
              result.primary_prediction
                .bonafide_probability *
              100
            ).toFixed(2)}
            %
          </p>

          <p>
            <strong>
              Spoof probability:
            </strong>{" "}
            {(
              result.primary_prediction
                .spoof_probability *
              100
            ).toFixed(2)}
            %
          </p>

          <hr />

          <h4>
            Model Comparison
          </h4>

          <div
            style={{
              padding: "1rem",
              border:
                "1px solid #ddd",
              borderRadius:
                "8px",
              marginBottom:
                "1rem",
            }}
          >
            <h5>
              CNN Baseline
            </h5>

            <p>
              <strong>
                Prediction:
              </strong>{" "}
              {result.cnn.label.toUpperCase()}
            </p>

            <p>
              <strong>
                Confidence:
              </strong>{" "}
              {(
                result.cnn.confidence *
                100
              ).toFixed(2)}
              %
            </p>

            <p>
              <strong>
                Spoof probability:
              </strong>{" "}
              {(
                result.cnn.spoof_probability *
                100
              ).toFixed(2)}
              %
            </p>
          </div>

          <div
            style={{
              padding: "1rem",
              border:
                "1px solid #ddd",
              borderRadius:
                "8px",
            }}
          >
            <h5>
              AST Model
            </h5>

            <p>
              <strong>
                Prediction:
              </strong>{" "}
              {result.ast.label.toUpperCase()}
            </p>

            <p>
              <strong>
                Confidence:
              </strong>{" "}
              {(
                result.ast.confidence *
                100
              ).toFixed(2)}
              %
            </p>

            <p>
              <strong>
                Spoof probability:
              </strong>{" "}
              {(
                result.ast.spoof_probability *
                100
              ).toFixed(2)}
              %
            </p>
          </div>

          <hr />

          <h4>
            Acoustic Features
          </h4>

          <p>
            <strong>
              Spectral centroid mean:
            </strong>{" "}
            {result.rir_features
              .spectral_centroid_mean
              .toFixed(2)}
          </p>

          <p>
            <strong>
              Spectral centroid std:
            </strong>{" "}
            {result.rir_features
              .spectral_centroid_std
              .toFixed(2)}
          </p>

          <p>
            <strong>
              Spectral flatness mean:
            </strong>{" "}
            {result.rir_features
              .spectral_flatness_mean
              .toExponential(3)}
          </p>

          <p>
            <strong>
              RMS energy std:
            </strong>{" "}
            {result.rir_features
              .rms_energy_std
              .toFixed(4)}
          </p>

          <hr />

          <h4>
            Segment Analysis
          </h4>

          {result.segments.length ===
          0 ? (
            <p>
              No segment results available.
            </p>
          ) : (
            <div>
              {result.segments.map(
                (segment, index) => (
                  <div
                    key={index}
                    style={{
                      marginBottom:
                        "0.75rem",
                      padding:
                        "0.75rem",
                      border:
                        "1px solid #ccc",
                      borderRadius:
                        "8px",
                    }}
                  >
                    <strong>
                      {segment.start_seconds.toFixed(
                        1,
                      )}
                      s -{" "}
                      {segment.end_seconds.toFixed(
                        1,
                      )}
                      s
                    </strong>

                    <p>
                      <strong>
                        Prediction:
                      </strong>{" "}
                      {segment.prediction.toUpperCase()}
                    </p>

                    <p>
                      <strong>
                        Spoof probability:
                      </strong>{" "}
                      {(
                        segment.spoof_probability *
                        100
                      ).toFixed(2)}
                      %
                    </p>

                    <p>
                      <strong>
                        Confidence:
                      </strong>{" "}
                      {(
                        segment.confidence *
                        100
                      ).toFixed(2)}
                      %
                    </p>

                    <p>
                      <strong>
                        Status:
                      </strong>{" "}
                      {segment.suspicious
                        ? "SUSPICIOUS"
                        : "NORMAL"}
                    </p>
                  </div>
                ),
              )}
            </div>
          )}

          <hr />

          <h4>
            Mel Spectrogram
          </h4>

          <img
            src={`${API_BASE_URL}${result.spectrogram_path}`}
            alt="Mel spectrogram"
            style={{
              width: "100%",
              maxWidth:
                "650px",
              display:
                "block",
              borderRadius:
                "8px",
              border:
                "1px solid #ccc",
            }}
          />

          <hr />

          <h4>
            Analysis History
          </h4>

          {history.length === 0 ? (
            <p>
              No previous analyses.
            </p>
          ) : (
            <div>
              {history.map(
                (item) => (
                  <div
                    key={item.id}
                    style={{
                      padding:
                        "0.75rem",
                      marginBottom:
                        "0.75rem",
                      border:
                        "1px solid #ddd",
                      borderRadius:
                        "8px",
                    }}
                  >
                    <strong>
                      {item.filename}
                    </strong>

                    <p>
                      Time:{" "}
                      {item.timestamp}
                    </p>

                    <p>
                      Model:{" "}
                      {item.model}
                    </p>

                    <p>
                      Prediction:{" "}
                      {item.prediction.toUpperCase()}
                    </p>

                    <p>
                      Confidence:{" "}
                      {(
                        item.confidence *
                        100
                      ).toFixed(2)}
                      %
                    </p>
                  </div>
                ),
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}