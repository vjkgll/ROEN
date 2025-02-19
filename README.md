# ROEN

## 目录

- data
  - Darknet
  - ISCXIDS2012
  - TrafficLabelling
- models
  - 2012
  - 2017
  - 2020
- network.py
- untils.py
- train_2012.py
- train_2017.py
- train_2020.py
- requirements.txt

----

## Introduction

![Alt text](image.png)

\ \ As network attacks become increasingly complex and covert, traditional port-based traffic classification methods are no longer sufficient to meet modern cybersecurity needs. Existing traffic monitoring technologies face significant shortcomings in addressing emerging network threats, particularly in multi-modal traffic recognition, hybrid attack pattern detection, and traffic analysis under feature loss conditions. The main issues include: existing models exhibit poor adaptability, limited robustness, and high resource consumption when dealing with complex, uncertain, or missing information, leading to difficulties in ensuring prediction accuracy and stability when confronted with unknown patterns, sample imbalance, high-dimensional feature spaces, or feature loss. To address these challenges, this paper proposes an innovative approach, the Recurrent Feature Fusion Evolution Network (ROEN), which simultaneously captures both explicit statistical features and implicit dynamic topological structures. By employing a dual-thread multi-level recurrent feature fusion mechanism, ROEN significantly enhances classification accuracy. It performs exceptionally well in multi-modal traffic recognition, hybrid attack pattern detection, and traffic analysis under feature loss conditions, effectively tackling challenges across different datasets. Experimental results show that ROEN outperforms state-of-the-art models in classification performance on multiple public datasets while reducing model parameters and improving efficiency and flexibility in real-world applications. Notably, ROEN introduces the concept of dual-thread global information fusion, achieving a significant breakthrough in graph representation learning, with AUC/Multi-class AUC prediction performance improving by nearly 54.55%. Overall, ROEN provides an efficient and robust solution for traffic classification problems in complex network environments.

### Network Architecture

![Alt text](image-1.png)

\ \ The proposed Recurrent Feature Fusion Evolution Network (ROEN) consists of an Implicit Feature Extractor, an Explicit Feature Extractor, and a Global Feature Fusion Engine, as shown in Fig.
The Explicit Feature Extractor includes a multi-layer Perceptron (MLP), Adaptive Extraction (SAFC), and Loop Optimization (LOT). Its role is to extract the temporal feature representation of traffic statistics cyclically adaptively. The Implicit Feature Extractor consists of a Multi-layer Perceptron (MLP), Topological Extraction (TET), and Motion Capture (MCS), which are designed to capture the dynamic changes in network topology structure. The Global Feature Fusion Engine 
utilizes the SAMLP module to dynamically and adaptively integrate both explicit and implicit features.

**Implicit Feature Extractor (IFE)**
\ \ The Implicit Feature Extractor (IFE) includes a multi-level nonlinear information perception mechanism and a dynamic topology evolution implicit information fusion mechanism. The multi-level nonlinear information perception mechanism uses a Multi-Layer Perceptron (MLP) to extract nonlinear features from node attributes, removing noise interference from the features. The dynamic topology evolution implicit information fusion mechanism employs a Graph Convolutional Network (GCN) layer to embed node interactions at different time points and propagate topological information, ensuring that each node has access to global information. Then, Long Short-Term Memory (LSTM) layers are used to transmit information across different time points, capturing the evolving patterns of dynamic topological structure changes.

**Explicit Feature Extractor (EFE)**
\ \ The Explicit Feature Extractor (EFE) includes a multi-level nonlinear information perception mechanism and a self-attention recurrent optimization explicit information fusion mechanism. The multi-level nonlinear information perception mechanism extracts nonlinear features from edge attributes using Multi-Layer Perceptron (MLP) layers, removing noise interference from the features. The self-attention recurrent optimization explicit information fusion mechanism adaptively extracts edge statistical features using self-attention fully connected layers while feeding the extracted edge statistics at different time points into Long Short-Term Memory (LSTM) layers. It allows the model to fuse past and present edge information cyclically. Based on the fusion results, the fusion parameters and the edge feature extraction patterns are continuously adjusted, ultimately realizing a recurrent optimization process that identifies the general patterns of high-dimensional statistical features of edge information.

**Global Feature Fusion Engine (GFF)**
\ \ The Global Feature Fusion Engine (GFE) takes the extracted node topological structure evolution patterns and the general statistical patterns of edge information and inputs them into a self-attention MLP layer for adaptive dynamic fusion of global information. Finally, the fused global feature representation is passed through an activation function to perform network traffic classification.

### **Dataset Description**
Although datasets such as `Darknet`, `ISCXIDS2012`, and `TrafficLabelling` are mentioned, there is no detailed description of the sources, sizes, formats, etc. If possible, you can add a brief introduction to the datasets, especially how they help validate the performance of the ROEN model.

For example:
```markdown
### Datasets
- **Darknet**: This dataset contains network data for both attack and normal traffic, suitable for evaluation of network intrusion detection.
- **ISCXIDS2012**: This dataset includes intrusion detection data from multiple real-world network environments and has been widely used in network security research.
- **TrafficLabelling**: This dataset is used for classifying and labeling network traffic.

#### Training the Model
To train the ROEN model on the TrafficLabelling dataset, you can run the following command:
```bash
python train_2017.py
```

## :page_with_curl: 开源协议

### Citation

If you use this work in your own research, please cite the following paper:
```bibtex
@article{ROEN2025,
  author = {L. Ren, S. Wang, S. Zhong, Y. Li, B. Tang},
  title = {ROEN: Universal Dynamic Topology-Adaptive Evolution Model for Multi-modal Mixed Network Traffic Detection},
  journal = {Computer Networks},
  year = {2025},
  note = {Submitted}
}
```

### Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Submit a pull request
