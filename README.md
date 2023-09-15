# Awesome Geometric Deep Learning
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green)](https://github.com/traincheckai/awesome-geometric-deep-learning/pulls) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![Stars](https://img.shields.io/github/stars/traincheckai/awesome-geometric-deep-learning?color=yellow)  ![Forks](https://img.shields.io/github/forks/traincheckai/awesome-geometric-deep-learning?color=blue&label=Fork)

A curated list of awesome Geometric Deep Learning software frameworks

### [CoGDL](https://github.com/THUDM/CogDL)

#### Description & Software Languages
- CogDL is a graph deep learning toolkit that simplifies training and comparing standard or bespoke models for tasks like node and graph classification. 
- The primary language is Python. 

#### Features
- Emphasizes efficiency through optimized operators for faster training and GPU memory conservation.
- Provides user-friendly operation with straightforward APIs for experiments and hyper-parameter tuning. 
- Extensible design simplifies the application of GNN models to novel scenarios.

#### Docs & Tutorials Quality
- CProvides documentation on the necessary requirements and installation. 
- Documentation with example code is also provided for usage. 

#### Popularity & Community
-  stars: >1,500
- forks: >300 
- contributors: >30 

#### Creation Date
- May 17, 2019

#### Maintenance
- Not Active

#### License
- MIT license

---

### [Deep Graph Library (DGL)](https://github.com/dmlc/dgl)

#### Description & Software Languages
- DGL is a library designed to simplify the implementation of graph neural networks (GNNs) and other graph-based machine learning algorithms. 
- The primary programming language is Python.

#### Features
- Graph data handling through a flexible and easy to use efficient graph data structure. 
- Offers a collection of pre-built GNN layers that can be used to create custom GNN models. 
- Implements a message-passing framework which makes it simple to define new GNN layers and models. 

#### Docs & Tutorials Quality
- Comprehensive documentation and user guides  are available along with a discussion forum and slack channel. 
- The official documentation is available at https://docs.dgl.ai/en/latest/.


#### Popularity & Community
- stars: >11,000 
- forks: >2000 
- contributors: >200  

#### Creation Date
- April 15, 2018

#### Maintenance
- Active

#### License
- Apache License 2.0

---
### [DeepLab2](https://github.com/google-research/deeplab2)

#### Description & Software Languages
- DeepLab2 is a versatile TensorFlow library for deep labeling tasks, employing advanced neural networks to assign predicted values to image pixels.
- It supports a range of tasks including semantic, instance, and panoptic segmentation, depth estimation, and video panoptic segmentation, and incorporates state-of-the-art research models.
- The primary programming language is Python. 

#### Features
- DeepLab2 includes recent, cutting-edge research models for deep labeling tasks.
- The library also features multiple deep learning models such as Panoptic-DeepLab, Axial-DeepLab, MaX-DeepLab, STEP (Motion-DeepLab), ViP-DeepLab, and kMaX-DeepLab.
- Please note that, as of its current release, all demos operate in CPU mode.

#### Docs & Tutorials Quality
- Comes with an efficient installation guide and instructions for dataset conversion to TFRecord.
- Provides various Colab notebooks for off-the-shelf inference with different checkpoints.
- Has a FAQ section for common issues and a list of maintainers for contact in case of more complex problems.

#### Popularity & Community
- stars: >900 
- forks: >300 
- contributors: >10 

#### Creation Date
- May 2, 2021

#### Maintenance
- Not Active

#### License
- Apache License 2.0

---

### [DeltaConv](https://github.com/rubenwiersma/deltaconv)

#### Description & Software Languages
- DeltaConv is a library designed for geometric deep learning on point clouds.
- Introduced in the SIGGRAPH 2022 paper by Ruben Wiersma and team, it employs anisotropic convolution, a crucial part of CNNs, for curved surfaces.
- The primary programming language is Python.

#### Features
- DeltaConv is specifically built to handle point cloud data, making it well-suited for tasks involving 3D shapes and structures. 
- Can learn both scalar and vector features using geometric operators. This allows for a nuanced understanding of the geometric structure in the data it's working with.

#### Docs & Tutorials Quality
- The documentation provides comprehensive instructions on installing the library.
- A detailed guide is provided to replicate the experiments mentioned in the SIGGRAPH 2022 paper, including how to use pre-trained weights.
- The library includes a FAQ section that addresses common queries about GPU memory requirements, training time optimization, using bash scripts on Windows, and the rendering of figures in the paper.
- Overall, the library seems well-documented with extensive instructions on installation, usage, testing, and visualization. 


#### Popularity & Community
- stars: >100 
- forks: >5 
- contributors: 1 

#### Creation Date
- June 30, 2023

#### Maintenance
- Active

#### License
- MIT

--- 

### [DGL-LifeSci](https://github.com/awslabs/dgl-lifesci)

#### Description & Software Languages
- DGL-LifeSci is a DGL-based Python library focused on applying graph neural networks in life science.
- The primary programming language is Python.

#### Features
- DGL-LifeSci offers a range of functionalities for processing and analyzing molecular graphs and biological networks, including methods for graph construction, featurization, evaluation, and a range of pre-trained models.
- The library also includes model architectures and training scripts to facilitate the application of deep learning on these graphs.

#### Docs & Tutorials Quality
- Includes detailed instructions for installing the library. 
- The library provides a command-line interface that can be used even by those without programming experience.
- A link to examples is provided in the text. 
- A slack channel for real-time discussion.
- The creators of DGL-LifeSci have made an effort to support their users and provide necessary learning materials.

#### Popularity & Community
- stars: >500 
- forks: >100 
- contributors: >20  

#### Creation Date
- April 19, 2020

#### Maintenance
- Not Active

#### License
- Apache License 2.0

---
### [ESCNN](https://github.com/QUVA-Lab/escnn)

#### Description & Software Languages
- QUVA Lab's escnn library is a PyTorch extension for creating E(n)-equivariant Steerable Convolutional Neural Networks.
- Unlike typical CNNs, it ensures invariance under 2D and 3D transformations, enhancing model data efficiency.
- The primary programming language is Python.

#### Features
- The library facilitates various feature field definitions, equivariant operations, and provides four specialized subpackages.

#### Docs & Tutorials Quality
- Well documented repo that is supplemented with comprehensive tutorials.  

#### Popularity & Community
- stars: >100 
- forks: >30 
- contributors: >5  

#### Creation Date
- March 20, 2022

#### Maintenance
- Active

#### License
- BSD Clear Licence

---

### [Fourier Features](https://github.com/tancik/fourier-feature-networks)

#### Description & Software Languages
- This library enhances Multilayer Perceptron (MLP) performance in low-dimensional problem domains, using Fourier feature mapping to learn high-frequency functions. 
- It counters MLP's failure in learning high frequencies and transforms the Neural Tangent Kernel (NTK) into an adjustable stationary kernel. This approach significantly improves MLP for computer vision and graphics tasks by selecting problem-specific Fourier features.
- The primary programming language is Python.

#### Features 
- Fourier Feature Mapping for high-frequency function learning.
- Neural Tangent Kernel (NTK) transformation for overcoming MLP spectral bias.
- Tunable kernel bandwidth for data flexibility.
- Problem-specific Fourier features selection to optimize MLP performance.
- Tools to enhance MLP's learning of complex 3D objects and scenes.

#### Docs & Tutorials Quality
- The library includes a demo IPython notebook as a reference and scripts for generating paper plots and tables.
- The repository provides resources for understanding and implementing Fourier features in MLPs for low-dimensional regression tasks relevant to computer vision and graphics.

#### Popularity & Community
- stars: >1000 
- forks: >100 
- contributors: 4 

#### Creation Date
- June 14, 2020

#### Maintenance
- Not Active

#### License
- MIT 

---
### [GeometricfLUX](https://github.com/FluxML/GeometricFlux.jl)

#### Description & Software Languages
- GeometricFlux is a geometric deep learning extension for the Julia-based Flux machine learning library. 
- It supports graph-structured data, is compatible with the JuliaGraphs ecosystem, and offers CUDA GPU acceleration. 
- It also integrates well with other Flux-compatible packages.
- The primary programming language is Julia.

#### Features 
- GeometricFlux supports CUDA GPU acceleration with CUDA.jl, and mini-batched training to leverage GPU advantages.
- It supports Message-passing and Graph Network architectures.
- Supports both static and variable graph strategies, with the latter being useful for training models over diverse graph structures.
- Integrates GNN benchmark datasets.
- Supports dynamic graph updates for advanced manifold learning.

#### Docs & Tutorials Quality
- The GeometricFlux library has a well-documented README file on its GitHub repository. 
- It provides a clear overview of the library, its features, and its compatibility with other packages.
- It also includes installation instructions and code examples for constructing a Graph Convolutional Network (GCN) layer and using it in a Flux model, which is helpful for users to get started.

#### Popularity & Community
- stars: >300 
- forks: >20 
- contributors: >10  

#### Creation Date
- March 31, 2019

#### Maintenance
- Not Active

#### License
- MIT

---
### [GeomLoss](https://github.com/jeanfeydy/geomloss)

#### Description & Software Languages
- GeomLoss is a Python library offering GPU-accelerated geometric loss functions, including Kernel norms, Hausdorff divergences, and Debiased Sinkhorn divergences. 
- Integrated with PyTorch, it supports weighted point clouds, density maps, and volumetric masks, with specialized solutions for various problem sizes. 
- The primary programming language is Python.

#### Features 
- GeomLoss is an advanced Python interface for Optimal Transport algorithms, featuring batchwise computations, linear memory usage for large problems, fast kernel truncation, log-domain stabilization of Sinkhorn iterations, efficient gradient computation, and support for unbalanced Optimal Transport.
- It also offers ε-scaling heuristic support, improving speed for 3D problems. 
- The library implements symmetric, positive definite divergences, suitable for measure-fitting applications. 

#### Docs & Tutorials Quality
- The GeomLoss library has comprehensive documentation and tutorials. 
- The website includes clearly labeled sections for getting started, mathematics and algorithms, the PyTorch API, and a gallery of examples. 
- Each feature of the library is explained in detail, and there's a concrete code example provided to demonstrate its usage.
- The tutorial section provides a step-by-step guide to using the library, and there are separate pages detailing the mathematical and algorithmic foundations of the library's functions. 
- This depth of explanation indicates a strong commitment to user education.
- The website also provides a reference to the original academic work, suggesting the documentation's credibility.
- There are links to related projects, adding a valuable resource for users seeking additional tools or comparative information.
- There's a built documentation guide for Google Cloud and Google Colab, which may be helpful for users working in these environments.

#### Popularity & Community
- stars: >400 
- forks: >50 
- contributors: >5  

#### Creation Date
- Feb 24, 2019

#### Maintenance
- Not Active

#### License
- MIT

---
### [Graphein](https://graphein.ai/)

#### Description & Software Languages
- Graphein is a package that enables the creation of graph-based protein representations, compatible with geometric deep learning libraries like NetworkX, PyTorch Geometric, and DGL. 
- It simplifies protein analysis using graph theory and deep learning methods.
- The primary programming language is Python.


#### Features 
- Compatibility with standard PyData formats and graph objects for popular deep learning libraries.
- Both programmatic API and a command-line interface for constructing graphs.
- Creation of protein graphs, including from the AlphaFold Protein Structure Database.
- Generation of molecular graphs from smiles strings and various file formats (.sdf, .mol2, .pdb).
- Construction of RNA graphs, protein-protein interaction graphs, and gene regulatory network graphs.

#### Docs & Tutorials Quality
- The Graphein library provides several installation methods, including pip and conda environments, and a Dockerfile for containerized setup. 
- Thorough tutorials on Protein, Molecules, RNA Graphs, PPI Networks, Gene Regulatory Networks are also included in the documentation along with various data sets. 
- Overall, the documentation is very comprehensive and user friendly.

#### Popularity & Community
- stars: >800 
- forks: >100 
- contributors: >20 

#### Creation Date
- Aug 25, 2019

#### Maintenance
- Active

#### License
- MIT

--- 
### [Graph Nets](https://github.com/deepmind/graph_nets)

#### Description & Software Languages
- The Graph Nets library is a DeepMind's product for constructing graph networks using TensorFlow and Sonnet. 
- Graph networks are a type of graph neural network that accepts a graph as input and provides an updated graph as output, operating on edge, node, and global-level attributes. This technology is designed to enhance the learning of relational inductive biases in deep learning.
- The primary programming language is Python.

#### Features 
- The library provides an interface for creating custom graph networks with desired configurations using simple functions.
 - It also includes Jupyter notebook demos illustrating how to generate, manipulate, and train graph networks on tasks like shortest path-finding, sorting, and physical predictions. 
 
#### Docs & Tutorials Quality
- The repository provides a comprehensive explanation of what graph networks are and how they work, including a link to an arXiv paper for more detailed information. 
- It clearly describes the installation process for different environments, accommodating both TensorFlow 1 and TensorFlow 2 users, along with GPU and CPU-specific instructions.
- Moreover, usage examples are given in the readme, showing how to construct a simple graph network module and connect it to data. 
- The library also includes several Jupyter notebook demos that teach users how to create, manipulate, and train graph networks. These demos can be run locally or in Google Colaboratory, providing flexibility for different users.

#### Popularity & Community
- stars: >5000 
- forks: >700 
- contributors: >10 

#### Creation Date
- Aug 26, 2018

#### Maintenance
- Not Active

#### License
- Apache License 2.0


---
### [Jraph](https://github.com/deepmind/jraph)

#### Description & Software Languages
- Jraph is a lightweight library developed by DeepMind for implementing graph neural networks in JAX, a numerical computing library. 
- It is designed to facilitate the construction and training of graph neural networks, without prescribing a specific way to write or develop these networks. 
- The primary programming language is Python.

#### Features 
- Jraph offers GraphsTuple data structures for graph representation, utilities for operations like batching datasets, JIT compilation support, and defining losses on partitions of inputs. 
- It also houses a 'zoo' of lightweight graph neural network models.

#### Docs & Tutorials Quality
- Jraph's documentation comprises a quick start guide, interactive Colabs, and examples demonstrating its use. It also supports pytorch data loading and distributed graph network implementation. 
- Installation is simple via pip or directly from GitHub.

#### Popularity & Community
- stars: >1000 
- forks: >50 
- contributors: >10 

#### Creation Date
- Oct 18, 2020

#### Maintenance
- Not Active

#### License
- Apache License 2.0

--- 
### [Karate Club](https://github.com/benedekrozemberczki/karateclub)

#### Description & Software Languages
- Karate Club is an extension library for NetworkX, providing unsupervised machine learning on graph structured data. 
- It incorporates cutting-edge techniques for network embedding and community detection, making it a comprehensive tool for small-scale graph mining research.
- The primary programming language is Python

#### Features 
- The library includes  a relevant Paper, a Promo Video, and External Resources. It supports multiple community detection and embedding methods from leading network science, data mining, artificial intelligence, and machine learning conferences, workshops, and journals.
- New graph classification datasets are accessible via SNAP, TUD Graph Kernel Datasets, and GraphLearning.io.

#### Docs & Tutorials Quality
- Installation is as simple as running a pip command. 
- The library is frequently updated, so casual upgrading is recommended. 
- A collection of use cases and synthetic examples are provided for model illustration.

#### Popularity & Community
- stars: >1900 
- forks: >200 
- contributors: >10 

#### Creation Date
- Dec 1, 2019

#### Maintenance
- Active

#### License
- GNU General Public License v3.0.

---
### [POT](https://github.com/PythonOT/POT)

#### Description & Software Languages
- The POT (Python Optimal Transport) library includes a feature for Graph Neural Network OT layers, and these layers are part of the Geometric Deep Learning toolbox. 
- Therefore, while the POT library itself is not solely focused on GDL, it provides tools that can be employed in the GDL context.
- The primary programming language is Python.

#### Features 
- The Python Optimal Transport (POT) library offers solvers for various optimal transport problems in signal and image processing, and machine learning. 
- Variety of generic and specialized OT solvers, machine learning-related solvers, compatibility with multiple data frameworks, and Graph Neural Network OT layers. 

#### Docs & Tutorials Quality
- Well-documented and comes with clear instructions on its installation and use. 
- Clear citation details in both standard and Bibtex formats.
- Installation guidelines for various environments (Pip, Anaconda), as well as dependency information.
- Easy-to-follow examples of short code snippets demonstrating various functionalities of the library, like computing Wasserstein distances and OT matrix, and calculating the Wasserstein barycenter.
- A link to additional in-depth examples and use-cases in the library's GitHub repository.
- Information on various channels for support and joining the development discussion.

#### Popularity & Community
- stars: >2000 
- forks: >400 
- contributors: >50 

#### Creation Date
- Oct 16, 2016

#### Maintenance
- Active

#### License
- MIT

---
### [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)

#### Description & Software Languages
- PyTorch Geometric is a library for deep learning on irregular input data, such as graphs, point clouds, and manifolds. 
- It is an extension of PyTorch, providing geometric deep learning primitives and network architectures. 
- The primary programming language is Python.

#### Features 
- Flexible and easy-to-use APIs for various graph-structured data tasks.
- High-performance message-passing and graph convolution operations.
- A wide range of implemented graph neural network (GNN) architectures.
- Ready-to-use data handling and processing for several benchmark graph datasets.
- Integration with other PyTorch libraries, such as PyTorch Lightning.

#### Docs & Tutorials Quality
- PyTorch Geometric has comprehensive documentation, including installation instructions, usage examples, and detailed descriptions of available models and methods. 
- The official documentation is available at https://pytorch-geometric.readthedocs.io/en/latest/. 
- The repository also contains several Jupyter notebooks with examples and tutorials.

#### Popularity & Community
- stars: >17,000 
- forks: >3000 
- contributors: >400 

#### Creation Date
- Oct 5, 2018

#### Maintenance
- Active

#### License
- MIT

---
### [PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)

#### Description & Software Languages
- PyTorch Geometric Temporal is a Python library extending PyTorch Geometric for dynamic and temporal geometric deep learning. - Detailed examples and notebooks are provided for ease of understanding and implementation.
- The primary programming language is Python.

#### Features 
- It offers methods for embedding and spatio-temporal regression, an easy-to-use dataset loader for dynamic and temporal graphs, GPU support, and benchmark datasets. 
- It interfaces well with PyTorch Lightning, allowing seamless CPU and GPU training.

#### Docs & Tutorials Quality
- PyTorch Geometric Temporal is comprehensively documented, with detailed tutorials, such as epidemiological forecasting and web traffic management case studies. 
- It provides easy-to-follow examples, including a recurrent graph convolutional network implementation. 
- The library includes a wide variety of temporal graph neural network methods, all with references to the original papers. 
- It also offers clear instructions for installation, testing, and dataset creation, thus catering to users at all levels.

#### Popularity & Community
- stars: >2000 
- forks: >300 
- contributors: >20 

#### Creation Date
- June 21, 2020

#### Maintenance
- Active

#### License
- MIT

---
### [Spektral](https://github.com/danielegrattarola/spektral)

#### Description & Software Languages
- Spektral is a Python library built on Keras and TensorFlow 2, specifically designed for graph deep learning. 
- It enables the creation of graph neural networks (GNNs) and is suitable for various applications such as user classification in social networks, prediction of molecular properties, or link prediction.
- The primary programming language is Python.

#### Features 
- Spektral implements numerous layers for graph deep learning and pooling. 
- Notable layers include Graph Convolutional Networks (GCN), Chebyshev convolutions, GraphSAGE, ARMA convolutions, Graph attention networks (GAT), and more. 
- Pooling layers include MinCut pooling, DiffPool, Top-K pooling, and Global pooling among others. 
- Additionally, Spektral provides utilities for graph manipulation and transformation.

#### Docs & Tutorials Quality
- Spektral supports Python 3.6 and above and is installable from PyPi or from its source on Github. 
- The Spektral 1.0 release introduces a new datasets API, a Loader class for handling graph batches, a transforms module for common operations on graphs, and other features. 
- Community contributions are encouraged and the contribution guidelines are available on the project's Github page.

#### Popularity & Community
- stars: >2000 
- forks: >300 
- contributors: >20 

#### Creation Date
- Jan 13, 2019

#### Maintenance
- Active

#### License
- MIT

--- 
### [StellarGraph](https://github.com/stellargraph/stellargraph) 

#### Description & Software Languages
- StellarGraph is a Python library developed for machine learning on graph-structured data. 
- It's based on TensorFlow 2, Keras, Pandas, and NumPy, and it supports a variety of graph types.
- The primary programming language is Python.

#### Features 
- The library offers a set of state-of-the-art algorithms for tasks like representation learning, classification, and link prediction, including GraphSAGE, HinSAGE, Attri2Vec, Graph ATtention Network (GAT), Graph Convolutional Network (GCN), Node2Vec, Metapath2Vec, and more. 

#### Docs & Tutorials Quality
- Installation can be done through pip or Anaconda, and there's also an option to install directly from the GitHub source. 
- The library also provides an extensive set of examples to help users get started.
- If used in research, citation is requested using a given BibTex entry.

#### Popularity & Community
- stars: >2800 
- forks: >400 
- contributors: >30 

#### Creation Date
- Aug 8, 2018

#### Maintenance
- Not Active

#### License
- Apache License 2.0
