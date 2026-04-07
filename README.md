# SPIN (Stirring Parameter Intelligent Navigator) / (Smart Parameter Intervention Navigator) 
This repository provides a lightweight inference pipeline for predicting BDO concentration from fermentation time-series data and recommending RPM adjustments when a decrease is expected.

The system uses trained **BiLSTM/Transformer models** and supports:
- One-step prediction using the latest time window
- Automatic detection of BDO decrease
- RPM optimization to maximize predicted BDO

---

## 📁 Project Structure
```
.
├── predict.py # Entry point
├── predict_BiLSTM.py # BiLSTM inference logic
├── predict_Transformer.py # Transformer inference logic
├── model_BiLSTM.py # BiLSTM model definition
├── model_Transformer.py # Transformer model definition
│
├── Models/
│ ├── ckpt_BiLSTM.keras
│ └── ckpt_Transformer.pt
│
├── Input/
│ ├── example_dynamic_data.csv
│ └── example_static_data.csv
```
---

## ⚙️ Requirements

- Python 3.8+
- Tensorflow
- PyTorch
- NumPy
- Pandas
---

## 🚀 Usage
run with BiLSTM model
```
python predict.py --model BiLSTM --dynamic-csv ./Input/example_dynamic_data.csv --static-csv ./Input/example_static_data.csv
```

run with Transformer model
```
python predict.py --model Transformer --dynamic-csv ./Input/example_dynamic_data.csv --static-csv ./Input/example_static_data.csv
```
