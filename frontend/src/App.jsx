import { useEffect, useState } from "react";
import axios from "axios";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState("");
  const [patientData, setPatientData] = useState(null);
  const [explanationData, setExplanationData] = useState(null);

  const [loading, setLoading] = useState(false);
  const [explanationLoading, setExplanationLoading] = useState(false);

  const [error, setError] = useState("");
  const [explanationError, setExplanationError] = useState("");

  // --------------------------------------------------
  // Load patient list
  // --------------------------------------------------
  useEffect(() => {
    axios
      .get(`${API_BASE}/api/patients`)
      .then((response) => {
        const patientList = response.data.patients || [];

        setPatients(patientList);

        if (patientList.length > 0) {
          setSelectedPatient(patientList[0].OASISID);
        }
      })
      .catch((err) => {
        console.error(err);
        setError("Could not connect to the NeuroFEDX backend.");
      });
  }, []);

  // --------------------------------------------------
  // Load patient predictions
  // --------------------------------------------------
  useEffect(() => {
    if (!selectedPatient) return;

    setLoading(true);
    setError("");
    setPatientData(null);

    axios
      .get(`${API_BASE}/api/patient/${selectedPatient}`)
      .then((response) => {
        setPatientData(response.data);
      })
      .catch((err) => {
        console.error(err);
        setError("Could not load patient data.");
        setPatientData(null);
      })
      .finally(() => {
        setLoading(false);
      });
  }, [selectedPatient]);

  // --------------------------------------------------
  // Load patient-specific SHAP explanation
  // --------------------------------------------------
  useEffect(() => {
    if (!selectedPatient) return;

    setExplanationLoading(true);
    setExplanationError("");
    setExplanationData(null);

    axios
      .get(`${API_BASE}/api/patient/${selectedPatient}/explanation`)
      .then((response) => {
        setExplanationData(response.data);
      })
      .catch((err) => {
        console.error(err);
        setExplanationError(
          "Could not load the patient-specific explanation."
        );
        setExplanationData(null);
      })
      .finally(() => {
        setExplanationLoading(false);
      });
  }, [selectedPatient]);

  // --------------------------------------------------
  // Prediction card
  // --------------------------------------------------
  const renderModelResult = (title, model) => {
    if (!model) return null;

    return (
      <div className="model-card">
        <h3>{title}</h3>

        <div className="prediction">
          <span>Prediction</span>
          <strong>{model.prediction}</strong>
        </div>

        <div className="confidence">
          Confidence: {(model.confidence * 100).toFixed(2)}%
        </div>

        <div className="correct">
          {model.correct ? "✓ Correct" : "✗ Incorrect"}
        </div>
      </div>
    );
  };

  // --------------------------------------------------
  // Modality contribution bars
  // --------------------------------------------------
  const renderModalityContribution = (modalities) => {
    if (!modalities || modalities.length === 0) {
      return (
        <div className="no-data">
          No modality contribution available.
        </div>
      );
    }

    return (
      <div className="modality-list">
        {modalities
          .filter((item) => item.modality !== "UNMAPPED")
          .map((item) => {
            const percentage = (item.contribution || 0) * 100;

            return (
              <div className="modality-item" key={item.modality}>
                <div className="modality-header">
                  <span>{item.modality}</span>
                  <strong>{percentage.toFixed(2)}%</strong>
                </div>

                <div className="progress-track">
                  <div
                    className="progress-fill"
                    style={{ width: `${Math.min(percentage, 100)}%` }}
                  />
                </div>
              </div>
            );
          })}
      </div>
    );
  };

  // --------------------------------------------------
  // Top SHAP features
  // --------------------------------------------------
  const renderTopFeatures = (features) => {
    if (!features || features.length === 0) {
      return (
        <div className="no-data">
          No feature explanation available.
        </div>
      );
    }

    return (
      <div className="feature-list">
        {features.map((item, index) => {
          const shapValue = Number(item.shap_value || 0);

          return (
            <div className="feature-card" key={`${item.feature}-${index}`}>
              <div className="feature-rank">
                #{index + 1}
              </div>

              <div className="feature-main">
                <div className="feature-title-row">
                  <strong>{item.feature}</strong>

                  <span
                    className={
                      shapValue >= 0
                        ? "shap-value positive"
                        : "shap-value negative"
                    }
                  >
                    {shapValue >= 0 ? "+" : ""}
                    {shapValue.toFixed(5)}
                  </span>
                </div>

                <div className="feature-value">
                  Patient value:{" "}
                  <strong>
                    {typeof item.feature_value === "number"
                      ? item.feature_value.toFixed(4)
                      : item.feature_value}
                  </strong>
                </div>

                <div
                  className={
                    shapValue >= 0
                      ? "feature-direction positive-text"
                      : "feature-direction negative-text"
                  }
                >
                  {item.direction === "toward_predicted_class"
                    ? "↑ Contributes toward predicted class"
                    : "↓ Contributes away from predicted class"}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // --------------------------------------------------
  // Complete explanation card
  // --------------------------------------------------
  const renderExplanation = (title, model) => {
    if (!model) return null;

    return (
      <div className="explanation-card">
        <div className="explanation-header">
          <div>
            <h3>{title}</h3>
            <p>
              Prediction:{" "}
              <strong>{model.prediction}</strong>
              {" • "}
              Confidence:{" "}
              <strong>
                {(model.confidence * 100).toFixed(2)}%
              </strong>
            </p>
          </div>
        </div>

        <div className="explanation-grid">
          <div className="explanation-section">
            <h4>Top Contributing Features</h4>

            {renderTopFeatures(model.top_features)}
          </div>

          <div className="explanation-section">
            <h4>Modality Contribution</h4>

            <p className="explanation-note">
              Contribution is calculated from the absolute SHAP
              values of the features belonging to each modality.
            </p>

            {renderModalityContribution(
              model.modality_contribution
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="app">
      {/* ------------------------------------------------ */}
      {/* Header */}
      {/* ------------------------------------------------ */}
      <header className="header">
        <div>
          <h1>NeuroFEDX</h1>
          <p>
            Federated Explainable Alzheimer's Disease Analysis
          </p>
        </div>
      </header>

      <main className="container">
        {/* ------------------------------------------------ */}
        {/* Patient selector */}
        {/* ------------------------------------------------ */}
        <section className="patient-selector">
          <label>Select Patient</label>

          <select
            value={selectedPatient}
            onChange={(event) =>
              setSelectedPatient(event.target.value)
            }
          >
            {patients.map((patient) => (
              <option
                key={patient.OASISID}
                value={patient.OASISID}
              >
                {patient.OASISID}
              </option>
            ))}
          </select>
        </section>

        {/* ------------------------------------------------ */}
        {/* Loading / errors */}
        {/* ------------------------------------------------ */}
        {loading && (
          <div className="message">
            Loading patient data...
          </div>
        )}

        {error && <div className="error">{error}</div>}

        {/* ------------------------------------------------ */}
        {/* Patient dashboard */}
        {/* ------------------------------------------------ */}
        {patientData && !loading && (
          <>
            {/* Patient summary */}
            <section className="patient-summary">
              <div>
                <span>Patient ID</span>
                <strong>{patientData.OASISID}</strong>
              </div>

              <div>
                <span>Diagnosis Client</span>
                <strong>
                  {patientData.diagnosis.client_id}
                </strong>
              </div>

              <div>
                <span>Severity Client</span>
                <strong>
                  {patientData.severity.client_id}
                </strong>
              </div>

              <div>
                <span>Rows</span>
                <strong>
                  {patientData.diagnosis.n_rows}
                </strong>
              </div>
            </section>

            {/* ------------------------------------------------ */}
            {/* Diagnosis predictions */}
            {/* ------------------------------------------------ */}
            <section className="section">
              <div className="section-title">
                <h2>Diagnosis Analysis</h2>

                <span className="badge">
                  True: {patientData.diagnosis.true_label}
                </span>
              </div>

              <div className="model-grid">
                {renderModelResult(
                  "Global Federated Model",
                  patientData.diagnosis.global
                )}

                {renderModelResult(
                  "Personalized Federated Model",
                  patientData.diagnosis.personalized
                )}

                {renderModelResult(
                  "Local Model",
                  patientData.diagnosis.local
                )}
              </div>
            </section>

            {/* ------------------------------------------------ */}
            {/* Diagnosis explanation */}
            {/* ------------------------------------------------ */}
            <section className="section">
              <div className="section-title">
                <h2>Diagnosis Explainability</h2>

                <span className="explain-badge">
                  TreeSHAP
                </span>
              </div>

              {explanationLoading && (
                <div className="message">
                  Generating patient-specific diagnosis explanation...
                </div>
              )}

              {explanationError && (
                <div className="error">
                  {explanationError}
                </div>
              )}

              {explanationData &&
                !explanationLoading && (
                  <div className="explanation-stack">
                    {renderExplanation(
                      "Global Federated Model",
                      explanationData.diagnosis.global
                    )}

                    {renderExplanation(
                      "Personalized Federated Model",
                      explanationData.diagnosis.personalized
                    )}
                  </div>
                )}
            </section>

            {/* ------------------------------------------------ */}
            {/* Severity predictions */}
            {/* ------------------------------------------------ */}
            <section className="section">
              <div className="section-title">
                <h2>Severity Analysis</h2>

                <span className="badge">
                  True: {patientData.severity.true_label}
                </span>
              </div>

              <div className="model-grid">
                {renderModelResult(
                  "Global Federated Model",
                  patientData.severity.global
                )}

                {renderModelResult(
                  "Personalized Federated Model",
                  patientData.severity.personalized
                )}

                {renderModelResult(
                  "Local Model",
                  patientData.severity.local
                )}
              </div>
            </section>

            {/* ------------------------------------------------ */}
            {/* Severity explanation */}
            {/* ------------------------------------------------ */}
            <section className="section">
              <div className="section-title">
                <h2>Severity Explainability</h2>

                <span className="explain-badge">
                  TreeSHAP
                </span>
              </div>

              {explanationLoading && (
                <div className="message">
                  Generating patient-specific severity explanation...
                </div>
              )}

              {explanationError && (
                <div className="error">
                  {explanationError}
                </div>
              )}

              {explanationData &&
                !explanationLoading && (
                  <div className="explanation-stack">
                    {renderExplanation(
                      "Global Federated Model",
                      explanationData.severity.global
                    )}

                    {renderExplanation(
                      "Personalized Federated Model",
                      explanationData.severity.personalized
                    )}
                  </div>
                )}
            </section>
          </>
        )}
      </main>
    </div>
  );
}

export default App;