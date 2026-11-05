# SmartFlow FL — Federated Traffic Flow Prediction & Communication Cost Analyzer (PS-003)

## 1. Problem statement
PS-003: Federated traffic-flow prediction with communication-cost evaluation.

## 2. Problem being solved
Traffic junctions hold local data that is costly or sensitive to centralise. We want a shared prediction model without moving raw data, and we want to know what that costs in communication.

## 3. Proposed solution
A Streamlit app that simulates 2–4 junctions training collaboratively with FedAvg, measures every byte exchanged, and compares against a centralized baseline.

## 4. Federated learning concept
Server sends the global model → each client trains on local data → clients return only parameters → server averages them (weighted by sample count) → repeat.

## 5. System architecture
```
            GLOBAL SERVER (FedAvg)
                   |
   Junction A  Junction B  Junction C  Junction D
   Local ML    Local ML    Local ML    Local ML   -> model updates only
```
Clients are **simulated on one computer**. The demo shows the architecture; it is not a guarantee of privacy/security.

## 6. Technologies
Python, Streamlit, Pandas, NumPy, scikit-learn (RandomForest baseline), Plotly.

## 7. Install
```
pip install -r requirements.txt
```
## 8. Run
```
streamlit run app.py
```
Suggested demo flow: Overview → Traffic Prediction (run centralized) → Federated Learning (start training) → Communication Cost → Model Performance. You can upload your own CSV in the sidebar (columns: timestamp, junction_id, vehicle_count, average_speed, occupancy, hour, day_of_week).

## 9. How communication cost is calculated
Each update is the model's parameter vector (9 weights + bias) as float32 bytes (`numpy.nbytes`). Per round: upload = Σ client updates, download = global model × clients, total = upload + download, cumulative = running sum. Centralized cost is the size of the raw training data sent once. FL is **not** assumed cheaper; the app reports what is measured.

## 10. Future improvements
Real networked clients, neural-network models, update compression/quantization, secure aggregation and differential privacy, FedProx for non-IID data, real sensor data.
