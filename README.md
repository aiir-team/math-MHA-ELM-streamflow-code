# Monthly Streamflow Modeling using Latest Meta-Heuristic Algorithms with Extreme Learning Machine: A Comparative Study

# Links:
1. Code
    https://github.com/aiir-team/math-MHA-ELM-streamflow-original-code
2. Paper:
    https://github.com/aiir-team/math-MHA-ELM-streamflow-paper


```code 

## Models
# Evolutionary-based
{"name": "GA-ELM", "class": "GaElm", "param_grid": ga_paras},  # Genetic Algorithm (GA)
A New Workload Prediction Model Using Extreme Learning Machine and Enhanced Tug of War optimization

{"name": "CRO-ELM", "class": "CroElm", "param_grid": cro_paras},  # Coral Reefs Optimization (CRO)
Daily global solar radiation prediction based on a hybrid Coral Reefs Optimization–Extreme Learning Machine approach

## Swarm-based
{"name": "AGTO-ELM", "class": "AgtoElm", "param_grid": agto_paras},  # Artificial Gorilla Troops Optimization (AGTO)
Bearing Fault Diagnosis Using Extreme Learning Machine Based on Artificial Gorilla Troops Optimizer

{"name": "ARO-ELM", "class": "AroElm", "param_grid": aro_paras},  # Artificial Rabbits Optimization (ARO)
An Efficient Model for Sentiment Analysis using Artificial Rabbits Optimized Vector Functional Link Network

{"name": "AVOA-ELM", "class": "AvoaElm", "param_grid": aro_paras},  # African Vultures Optimization Algorithm (AVOA)
A novel three-stage hybrid learning paradigm based on a multi-decomposition strategy, optimized relevance vector machine, and error correction for multi-step forecasting of precious metal prices

{"name": "DMOA-ELM", "class": "DmoaElm", "param_grid": dmoa_paras},  # Dwarf Mongoose Optimization Algorithm (DMOA)

{"name": "HGS-ELM", "class": "HgsElm", "param_grid": hgs_paras},  # Hunger Games Search (HGS)
Underwater Backscatter Recognition Using Deep Fuzzy Extreme Convolutional Neural Network Optimized via Hunger Games Search

{"name": "SSA-ELM", "class": "SsaElm", "param_grid": ssa_paras},  # Sparrow Search Algorithm (SSA)
Research on diagnosis method of series arc fault of three-phase load based on SSA-ELM

{"name": "WOA-ELM", "class": "WoaElm", "param_grid": woa_paras},  # Whale Optimization Algorithm (WOA)
Extreme learning machine optimized by whale optimization algorithm using insulated gate bipolar transistor module aging degree evaluation

# ## Physics-based
{"name": "NRO-ELM", "class": "NroElm", "param_grid": nro_paras},  # Nuclear Reaction Optimization (NRO)
A comprehensive comparison of recent developed meta-heuristic algorithms for streamflow time series forecasting problem

{"name": "HGSO-ELM", "class": "HgsoElm", "param_grid": hgso_paras},  # Henry Gas Solubility Optimization (HGSO)
Extreme learning machine evolved by fuzzified hunger games search for energy and individual thermal comfort optimization

{"name": "ASO-ELM", "class": "AsoElm", "param_grid": aso_paras},  # Atom Search Optimization (ASO)
Integrated framework of extreme learning machine (ELM) based on improved atom search optimization for short-term wind speed prediction

{"name": "ArchOA-ELM", "class": "ArchoaElm", "param_grid": archoa_paras},  # Archimedes Optimization Algorithm (ArchOA)


# # Human-based added
{"name": "CHIO-ELM", "class": "ChioElm", "param_grid": chio_paras},  # Coronavirus Herd Immunity Optimization (CHIO)

{"name": "LCO-ELM", "class": "LcoElm", "param_grid": lco_paras},  # Life Choice-based Optimization (LCO)
An Improved Sea Lion Optimization for Workload Elasticity Prediction with Neural Networks

{"name": "TLO-ELM", "class": "TloElm", "param_grid": tlo_paras},  # Teaching Learning-based Optimization (TLO)
Comparison of modified teaching–learning-based optimization and extreme learning machine for classification of multiple power signal disturbances

# ## Bio-based
{"name": "SMA-ELM", "class": "SmaElm", "param_grid": sma_paras},  # Slime Mould Algorithm (SMA)
{"name": "SOA-ELM", "class": "SoaElm", "param_grid": soa_paras},  # Seagull Optimization Algorithm (SOA)

{"name": "TSA-ELM", "class": "TsaElm", "param_grid": tsa_paras},  # Tunicate Swarm Algorithm (TSA)
Short‐term commercial load forecasting based on peak‐valley features with the TSA‐ELM model

# ## Music-based group
{"name": "HS-ELM", "class": "HsElm", "param_grid": hs_paras},  # Harmony Search (HS)
A hybrid intelligent model for medium-term sales forecasting in fashion retail supply chains using extreme learning machine and harmony search algorithm


# ## System-based
{"name": "AEO-ELM", "class": "AeoElm", "param_grid": aeo_paras},  # Artificial Ecosystem-based Optimization (AEO)

# ## Math-based
{"name": "GBO-ELM", "class": "GboElm", "param_grid": gbo_paras},  # Gradient-Based Optimizer (GBO)
Extreme Learning Machine Using Improved Gradient-Based Optimizer for Dam Seepage Prediction

{"name": "PSS-ELM", "class": "PssElm", "param_grid": pss_paras},  # Pareto-like Sequential Sampling (PSS)

{"name": "INFO-ELM", "class": "InfoElm", "param_grid": info_paras},  # weIghted meaN oF vectOrs (INFO)

{"name": "RUN-ELM", "class": "RunElm", "param_grid": run_paras},  # RUNge Kutta optimizer (RUN)
```



## How to run code
1. Run MLP model by: mlp.py 
2. Run Rnn-based model (RNN, LSTM) by: script_traditional_rrn_based.py (multiprocessing) 
3. All hybrid-MLP model by: script_hybrid_mlp.py (multiprocessing also)
4. Get the results table mean, std, cv by: get_table_results.py
5. Results saved in folder: history/results/ based on daily or weekly datatype
 
 
# Results in paper
- Convergence errors of models in docs/results_of_paper/final_results_from_Thieu.xlsx
- Stability figures in the paper in docs/results_of_paper/images/...

## Environment

1. Nếu sử dụng conda hoặc miniconda quản lí thư viện
```code 
conda create -n new ml python==3.8.5
conda activate ml

pip install numpy, pandas, scikit-learn, matplotlib, seaborn, h5py
pip install xlsxwriter==3.0.3, openpyxl==3.0.9
pip install tensorflow==2.9.1
pip install keras==2.9.0
pip install permetrics==1.3.2
pip install mealpy==2.5.1
```

2. Nếu sử dụng pip để quản lí thư viện 

2.1 Trên windows

```code 
open terminal: (pip on windows is already installed inside python)
   python -m venv ml
   ml\Scripts\activate.bat
   pip install -r requirements.txt
   
Or using Pycharm, setting for this environment and install packages in pycharm terminal
```

2.2 Trên linux/ubuntu

```code 
sudo apt-get install pip3
sudo apt-get install python3.8-venv

python3 -m venv ml 
source ml/bin/activate
pip install -r requirements.txt
```

## Kiểu loại dataset

- Nếu chỉ có 1 file dataset thì thiết lập tham số: 
    + SINGLE_CSV = True trong file config.py và config_kfold.py
    + NAME_DATASET: Tên file dataset, hệ thống sẽ tự động chia train, test hoặc validation 

- Nếu có 2 hoặc 3 file datasets đã chia sẵn train, validation, test thì thiết lập tham số:
    + SINGLE_CSV = False 
    + NAME_TRAIN_DATASET = "tên file train.csv"
    + NAME_VALID_DATASET = "tên file validation.csv"
    + NAME_TEST_DATASET = "tên file test.csv"


## Run normal model: train-validation-test 

```code 

1. Check file config.py cho các models loại này 

2. Chạy file: script_normal_elm.py cho model ELM truyền thống 

3. Chạy file: script_normal_mha_elm.py cho các model MHA-ELM 

4. Chạy file: script_normal_model_figures.py để lấy các hình vẽ cho các models
    + Cần bỏ comment model ELM trong file config, thì mới vẽ hình được và tính statistics được 
    
5. Chạy file: script_normal_summary_statistics.py để láy các metrics statatistics như min, max, mean sau N lần chạy 

6. Chạy file: script_get_weights.py để lấy weights của từng model hoặc lấy weights của model tốt nhất 
    + Model tốt nhất được đánh giá bằng hàm fitness dựa vào metrics đã chọn và weight đã chọn trong file config.py 
    
    PHASE1_BEST_METRICS = ["RMSE_train", "MAPE_train", "RMSE_test", "MAPE_test"]
    PHASE1_BEST_WEIGHTS = [0.1, 0.1, 0.6, 0.2]

    fitness = 0.1 * RMSE_train + 0.1 * MAPE_train + 0.6 * RMSE_test + 0.2 * MAPE_test
    
    + Nếu tất cả metrics đều là giá trị min tốt nhất thì model nào cho fitness bé nhất là tốt nhất
    + Nết tát cả metrics đều là giá trị max tốt nhất thì model nào cho fitness to nhất là tốt nhất  
    
7. Chạy file: script_normal_validate_real_world.py để validate những model đã train với dữ liệu thực tế 
- Cần phải thay đổi dataset phù hợp trong file này 

```


## Run k-fold model: cross-validation and test 

```code 

1. Config file for all k-fold models in: config_kfold.py

2. Run traditional ELM with k-fold with: script_kfold_elm.py

3. Run hybrid-ELM with k-fold with: script_kfold_mha_elm.py

    + Tất cả kết quả sẽ lưu ở đường dẫn: data/results_cv/tên_models/...
    + Kết quả sẽ lưu ở dạng file:  ...-metrics_all.csv. Đây là kết quả kfold-crossvalidation 
    + Mỗi model sẽ có thêm file: metric-final.xlsx chứa các sheet name như min, max, mean, best. Đây là giá trị tổng hợp cuối cùng cho từng model, và từng bộ tham số. 
    + Còn kết quả re-train với bộ tham số tốt nhất sẽ lưu ở: data/results_cv/retrain/

Ví dụ: PSO-ELM với các tham số w1=[0.2, 0.4], w2=[0.8, 1.0].
Thì số models tạo ra từ PSO-ELM là 4 với tham số như sau:
PSO (w1=0.2, w2=0.8)
PSO (w1=0.2, w2=1.0)
PSO (w1=0.4, w2=0.8)
PSO (w1=0.4, w2=1.0)

Sau đó nó sẽ cho ra kết quả bộ tham số tốt nhất dựa vào metrics đã chọn trong file: metric-final.xlsx (sheet = best)
Ở đây sẽ dựa vào hàm fitness: với weights kèm theo các metrics trong file: config_kfold.py 

BEST_MODEL_METRIC_TUNE_PARAS = ["RMSE_test", "MAPE_test"]
BEST_MODEL_WEIGHT_TUNE_PARAS = [0.5, 0.5]

fitness = 0.5 * RMSE_test + 0.5 * MAPE_test 

+ Nếu tất cả metrics đều là giá trị min tốt nhất thì model nào cho fitness bé nhất là tốt nhất
+ Nết tát cả metrics đều là giá trị max tốt nhất thì model nào cho fitness to nhất là tốt nhất  

Bình thường ở các thư viện nó chỉ cho chọn dựa vào 1 metric, tức là dựa vào RMSE hoặc MAPE để chọn bộ tham số tốt nhất.
Nếu muốn vậy chỉ cần thiết lập weight = 0 hoặc không thiết lập là được. 

    
4. Chạy file: script_kfold_summary_statistics.py để lấy các thông tin về statistics cho từng model đã retrain 

5. Chạy file: script_kfold_get_model_figures.py để vẽ hình cho từng model đã retrain

6. Chạy file: script_kfold_validate_real_world_cv.py để validate những model với bộ tham số tốt nhất (crossvalidation) 
- Cần lựa chọn đường dẫn vào bộ data thực tế 

- Nên nhớ là ở đây chỉ validate những models có bộ tham số tốt nhất, vì sau quá trình cross-validation, thì mỗi 
model chỉ có 1 bộ tham số tốt nhất. 

```


- Để lấy dataset cho từng loại chạy thì chạy file: script_get_experiment_dataset.py 
- Comment vào từng hàm nếu không muốn lấy dataset của loại chạy đó

- Để lấy weights của model hoặc model tốt nhất thì chạy file: script_get_weights.py
```code 
+ Model tốt nhất được đánh giá bằng hàm fitness dựa vào metrics đã chọn và weight đã chọn trong file config_kfold.py 
    
  BEST_MODEL_METRIC_RETRAIN = ["RMSE_train", "MAPE_train", "RMSE_test", "MAPE_test"]
  BEST_MODEL_WEIGHT_RETRAIN = [0.1, 0.1, 0.6, 0.2]
    
  fitness = 0.1 * RMSE_train + 0.1 * MAPE_train + 0.6 * RMSE_test + 0.2 * MAPE_test
    
  + Nếu tất cả metrics đều là giá trị min tốt nhất thì model nào cho fitness bé nhất là tốt nhất
  + Nết tát cả metrics đều là giá trị max tốt nhất thì model nào cho fitness to nhất là tốt nhất  

```


## Run phase 2 algorithms

- Có 3 loại chạy giải thuật với phase 2.

1. Tìm models tốt nhất trong các model normal (train-validation-test) và chạy phase 2 với model tốt nhất đó
2. Tìm models tốt nhất trong các model k-fold (cross-validation-test) và chạy phase 2 với model tốt nhất đó
3. Chọn bất kì 1 model nào ở phase 1 (cả 2 loại đều được: normal hoặc k-fold), xong rồi chạy phase 2 cho model đã chọn


```code 

1. Config file for all phase2 algorithms in: config_p2.py

Trong này có cài đặt tương ứng theo 3 cách chạy bên trên với tham số: PROJECT = "normal"
    + normal: là chạy cho project 1 (train-validation-test)
    + cv: là chạy cho project 2 (cross-validation-test)
    + select: là chạy cho bất kì 1 model nào đó. Nếu thiết lập tham số này thì phải thiết lập thêm tham số bên dưới 
    
    ## This is used when PROJECT = "select"
    SELECTED_MODEL = {
        "model": "CRO-ELM",                                             # Tên mô hình phase 1 được chọn 
        "folder": f"{DATA_DIRECTORY}/results/trial-0/CRO-ELM",          # Đường dẫn đến mô hình đó
        "file": "50-50-0.85-0.9-0.1-0.1-0.1-(0.02, 0.2)-0.1-3-model",   # Tên file của mô hình đó 
        "project": "normal"                                             # Nó thuộc project nào 
    }


2. Run all algoirthms with: script_p2_mha.py

- File này sẽ chạy các giải thuật được chọn trong file: config_p2.py và cách chạy projec nào 

3. Chạy file: script_p2_model_figures.py

- Chạy file này để vẽ hình fitness values (convergence chart)

4. Chạy file: script_p2_summary_statistics.py

- Chạy file này để tính statistics cho các giải thuật sau N lần chạy 

5. Chạy file: script_p2_get_weights.py

- Các giải thuật phase 2 không có weights, chạy file này hiển thị ra 1 vài thông tin cơ bản 


```




## MHA-ELM

- ELM: Extreme Learning Machine

- Mô hình này ban đầu cũng là mạng cải tiến của mô hình MLP. Có thể có 1 hoặc nhiều tầng ẩn. Tuy nhiên với 1 tầng ẩn là
  quá đủ để mapping tất cả các thể loại hàm yêu cầu.
- Cơ chế và sự khác biệt giữa MLP và ELM chỉ là cách huấn luyện mạng nơ-ron như sau:

    + Weights giữa tầng input và tầng hidden được random ngẫu nhiên, và sẽ không thay đổi.
    + Wegiths giữa tầng hidden và tầng output theo MLP chính thống sẽ là được tính mà cách nhân ma trận. Vì khi đã biết
      được giá trị thực tế, labels y. Thì chỉ cần nhân ma trận là tìm ra weights. Đặc biệt với cách này có thể tìm chính
      xác được điểm global optimal. Tuy nhiên vấn đề là nó chỉ là global optimal cho cái bộ random weight bên trên. Với
      bộ random weight khác nó lại là global optimal khác.

- Ưu điểm của ELM:

    + Fast, cực kì nhanh, không dùng gradient descent để train mạng. Không có epochs/generations --> Train bộ dữ liệu
      cực kì lớn cũng chỉ vài giây vì nó thực chất chỉ là nhân ma trận thôi.
    + Hiệu quả cao vì nhân ma trận để tính được global optimal, nhưng cần phải xét số lượng hidden unit lớn, như vậy mới
      học được dữ liệu lớn. Ví dụ: 1 triệu data --> 100000 hidden size chẳng hạn. Thì kết quả mới tốt.

- Nhược điểm của ELM:

    + Như đã giải thích bên trên, nhân ma trận chỉ tìm ra được global optimal cho bộ random weight đó. Vậy nếu bộ random
      weight nó xấu, thì global optimal tìm được chưa chắc đã là global optimal cho toàn bài toán.

- Cải tiến ELM dùng Metaheuristics:

    + Dùng Metaheuristics để tạo ra 1 quẩn thể (tập) các solution. Mỗi solution là 1 bộ random weight giữa tầng input và
      tầng hidden. Còn weights giữa tầng hidden và tầng output ta vẫn dùng nhân ma trận như thường.
    + Vậy có thể thấy, Metaheuristics giúp mạng ELM xét qua nhiều bộ random weight hơn --> Chăc chắn sẽ tốt hơn ELM
      thường.
    + Đặc điểm nữa là: Khi dùng Metaheuristics, ta không cần thiết lập số hidden size lớn, vì ta đã duyệt qua nhiều khả
      năng, nên chỉ cần set nhỏ như mạng bình thường là được.

    + Nhược điểm của cải tiến này là: Thời gian huấn luyện sẽ lâu hơn ELM thường, vì metaheuristics cũng sử dụng
      epochs/generations y như neural network vậy.

- Một số cách cải tiến khác của ELM:

    + Một số bài báo dùng Kernel ELM, tức là thay thế việc random weight của tầng input và tầng hidden bằng các kĩ thuật
      khác ví dụ kĩ thuật tính khoảng cách từ mạng Radial Basic Function Network. Còn weights giữa tầng hidden và tầng
      output:
        + Nếu sử dụng Gradient Descent thì nó sẽ trở thành mạng Radial Basic Function Network.
        + Nếu sử dụng nhân ma trận như ELM thường thì là 1 biến thể của ELM và có thể sẽ tốt hơn.


- Lý thuyết là vậy. Ta đi chứng mình bằng thực nghiệm.

<img src="data/img/elm1.png" alt="ELM" width="500" />

![image info](./data/img/elm2.png)




## Result files
```code 
+ folder: results/{test_size}-{m-rules}/trial-{x}/model-name/
    + test size trước anh để cố định 0.3, giờ anh để bao nhiêu cũng được
    + m-rules anh để mặc định là 15, giờ bao nhiêu cũng được
    + trial-{x} là số lần mình thử nghiệm với mỗi model, và mỗi cấu hình của model. Vì thuật toán MHO cho ra kết quả 
    khác nhau sau mỗi lần, do vậy mình thường chạy nhiều lần, rồi về so tính giá trị mean, STD, CV...
    + model-name là tên model ví dụ như GA-ANFIS, bên trong kết quả của mỗi folder model-name sẽ có 2 folder con là 
    model và visual.
        + model: là lưu kết quả của mô hình ví dụ như prediction, loss trong khi train, và các metrics tính toán
        + visual sẽ lưu các hình vẽ cho lần chạy đó.
+ filename: loss_train-10-50-0.85-0.1: tham số của GA trong file config.

```

## Notes

https://stackoverflow.com/questions/2186525/how-to-use-glob-to-find-files-recursively
https://stackoverflow.com/questions/14509192/how-to-import-functions-from-other-projects-in-python
https://stackoverflow.com/questions/51520/how-to-get-an-absolute-file-path-in-python
https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time


https://datatofish.com/read_excel/

https://pythontic.com/modules/pickle/dumps
https://medium.com/fintechexplained/how-to-save-trained-machine-learning-models-649c3ad1c018
https://github.com/keras-team/keras/issues/14180
https://ai-pool.com/d/how-to-get-the-weights-of-keras-model-

```python 
https://stackoverflow.com/questions/1894269/how-to-convert-string-representation-of-list-to-a-list

import json 
x = "[0.7587068025868327, 1000.0, 125.3177189672638, 150, 1.0, 4, 0.1, 10.0]"
solution = json.loads(x)
print(solution)

```