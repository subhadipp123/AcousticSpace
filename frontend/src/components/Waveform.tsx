import { useEffect, useRef } from "react";

import WaveSurfer from "wavesurfer.js";

import RegionsPlugin from
  "wavesurfer.js/dist/plugins/regions.esm.js";


interface WaveformSegment {
  start_seconds: number;
  end_seconds: number;
  prediction: "bonafide" | "spoof";
  confidence: number;
  bonafide_probability: number;
  spoof_probability: number;
  suspicious: boolean;
}


interface WaveformProps {
  audioUrl: string;
  segments?: WaveformSegment[];
}


export function Waveform({
  audioUrl,
  segments = [],
}: WaveformProps) {
  const containerRef =
    useRef<HTMLDivElement | null>(
      null,
    );


  useEffect(() => {
    if (!containerRef.current) {
      return;
    }

    const regionsPlugin =
      RegionsPlugin.create();


    const wavesurfer =
      WaveSurfer.create({
        container:
          containerRef.current,

        height: 120,

        normalize: true,

        waveColor: "#7c83fd",

        progressColor:
          "#4f46e5",

        cursorColor:
          "#111827",

        plugins: [
          regionsPlugin,
        ],
      });


    wavesurfer.load(
      audioUrl,
    );


    wavesurfer.on(
      "ready",
      () => {
        for (
          const segment
          of segments
        ) {
          if (
            !segment.suspicious
          ) {
            continue;
          }


          regionsPlugin.addRegion({
            start:
              segment.start_seconds,

            end:
              segment.end_seconds,

            drag: false,

            resize: false,

            color:
              "rgba(255, 80, 80, 0.22)",
          });
        }
      },
    );


    return () => {
      wavesurfer.destroy();
    };

  }, [
    audioUrl,
    segments,
  ]);


  return (
    <div
      style={{
        marginTop: "1rem",
        padding: "1rem",
        border:
          "1px solid #ccc",
        borderRadius:
          "8px",
      }}
    >
      <h4>
        Waveform & Suspicious Regions
      </h4>

      <div
        ref={
          containerRef
        }
      />

      <p
        style={{
          fontSize:
            "0.85rem",
          marginTop:
            "0.75rem",
        }}
      >
        Highlighted regions
        indicate suspicious
        audio segments.
      </p>
    </div>
  );
}