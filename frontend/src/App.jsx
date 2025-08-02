
import React, { useState } from "react";

export default function App() {
  const [file, setFile] = useState(null);
  const [output, setOutput] = useState(null);

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    formData.append("target_class", "golden retriever");

    const res = await fetch("http://localhost:8000/spoof/?epsilon=0.05", {
      method: "POST",
      body: formData,
    });

    const blob = await res.blob();
    setOutput(URL.createObjectURL(blob));
  };

  return (
    <div className="max-w-2xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Adversarial Image Tool</h1>
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <button
        onClick={handleUpload}
        className="mt-4 px-4 py-2 bg-blue-600 text-white rounded"
      >
        Apply Spoof
      </button>

      {output && (
        <div className="mt-6">
          <h2 className="font-semibold mb-2">Manipulated Image:</h2>
          <img src={output} alt="Output" className="border shadow" />
        </div>
      )}
    </div>
  );
}
