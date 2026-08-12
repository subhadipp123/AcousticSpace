import { AudioUpload } from "./components/AudioUpload";
import "./App.css";

function App() {
  return (
    <div style={{ padding: "2rem" }}>
      <h1>AcousticSpace</h1>
      <p>Deepfake audio detection via Room Impulse Response analysis</p>
      <AudioUpload />
    </div>
  );
}

export default App;