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
├── predict.py # Entry point (model dispatcher)
├── predict_Transformer.py # Transformer inference logic
├── predict_BiLSTM.py # (empty placeholder)
├── model_Transformer.py # Transformer model definition
├── model_BiLSTM.py # (empty placeholder)
├── ckpt_transformer.pt # Pretrained Transformer checkpoint
├── example_dynamic_data.csv # Example dynamic input
├── example_static_data.csv # Example static input
---
```

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
python predict.py --model BiLSTM --dynamic-csv example_dynamic_data.csv --static-csv example_static_data.csv
```

run with Transformer model
```
python predict.py --model Transformer --dynamic-csv example_dynamic_data.csv --static-csv example_static_data.csv
```
