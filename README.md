# GEnSHIN: Graph Enhanced Spatio-temporal Hierarchical Inference Network

## 📌 Overview
**GEnSHIN** (Graph Enhanced Spatio-temporal Hierarchical Inference Network) is a novel deep learning architecture designed for traffic flow forecasting in intelligent transportation systems. The model effectively captures complex spatiotemporal dependencies in traffic networks through innovative graph-based enhancements and hierarchical reasoning mechanisms.

## 🎯 Key Features
- **Attention-Enhanced GCRU Units**: Integrates Transformer modules with Graph Convolutional Recurrent Units to capture long-term temporal dependencies.
- **Asymmetric Dual-Embedding Graph Generation**: Learns data-driven asymmetric topological structures while preserving real road network information.
- **Dynamic Memory Bank**: Stores learnable traffic pattern prototypes and provides personalized representations for each sensor node.
- **Adaptive Graph Updater**: Dynamically adjusts graph structures during decoding to adapt to changing network conditions.

## 🏗️ Model Architecture
GEnSHIN consists of three main components:
1. **Encoder**: Processes historical traffic data using attention-enhanced GCRU units and Transformer modules.
2. **Memory Bank Module**: Stores traffic pattern prototypes and retrieves relevant patterns through attention mechanisms.
3. **Decoder**: Generates predictions using enhanced representations and dynamically updated graph structures.

## 📊 Performance
The model has been evaluated on the METR-LA dataset and demonstrates state-of-the-art performance in traffic flow prediction:

| Metric | Performance |
|--------|-------------|
| MAE | 3.60 |
| RMSE | 7.69 |
| MAPE | 9.06% |

GEnSHIN shows particularly strong performance during peak traffic hours (morning and evening rush periods), maintaining stable and accurate predictions under challenging conditions.

## 🔬 Technical Highlights
- **Non-Symmetric Graph Learning**: Captures directional traffic flow patterns that often exhibit asymmetry (e.g., inbound vs. outbound traffic during rush hours).
- **Pattern-Aware Representations**: Each sensor node receives personalized traffic pattern embeddings based on its historical behavior.
- **Multi-Scale Spatiotemporal Modeling**: Combines local spatial dependencies with global temporal patterns through hierarchical processing.

## 📈 Applications
GEnSHIN is suitable for various intelligent transportation applications:
- Urban traffic flow prediction
- Congestion forecasting
- Route optimization
- Traffic management systems
- Smart city infrastructure planning

## 📚 Citation
If you use GEnSHIN in your research, please cite our work:

```bibtex
@article{zhou2024genshin,
  title={GEnSHIN: Graph Enhanced Spatio-temporal Hierarchical Inference Network for Traffic Flow Forecasting},
  author={Zhou, Zhiyan and Liao, Junjie and Zhang, Wenhao and Liao, Yingyi and Wang, Ziai},
  journal={arXiv preprint},
  year={2024}
}
```

## 🔗 Repository
The complete implementation of GEnSHIN is available in this repository. The code provides:
- Model architecture implementation
- Data preprocessing utilities
- Evaluation metrics
- Pre-trained model weights (where applicable)

## 📄 License
This project is released under the MIT License. See the LICENSE file for details.

## 🤝 Contributing
We welcome contributions from the research community. Please feel free to submit issues, feature requests, or pull requests to help improve GEnSHIN.

## 🙏 Acknowledgments
We thank the Beijing Normal University (Zhuhai) Supercomputing Center for providing computational resources. We also acknowledge all researchers whose previous work has contributed to the development of this model.

---

*Note: This implementation is for research purposes. For production deployment in critical transportation systems, additional validation and safety considerations are recommended.*
