```bash
conda create -n Style-Bert-VITS2 python=3.10

pip install -r requirements.txt
# pip install nvidia-cublas-cu11 nvidia-cudnn-cu11  # for faster-whisper
# apt install libcublas11
python initialize.py

mkdir -p inputs/lain
gdown --fuzzy https://drive.google.com/file/d/1UDIUnpbar1CPYk1Ayl5X-SkIHyIjlqUO/view?usp=sharing -O inputs/lain.4.1.tar.gz
tar zxvf inputs/lain.4.1.tar.gz -C inputs/lain/


dataset_root="Data"
assets_root="model_assets"
input_dir="inputs/lain"
model_name="lain"

python slice.py -i ${input_dir} -o ${dataset_root}/${model_name}/raw

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/peng/anaconda3/envs/Style-Bert-VITS2/lib/python3.10/site-packages/nvidia/cudnn/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/peng/anaconda3/envs/Style-Bert-VITS2/lib/python3.10/site-packages/nvidia/cublas/lib/

python transcribe.py -i ${dataset_root}/${model_name}/raw -o ${dataset_root}/${model_name}/esd.list --speaker_name ${model_name} --compute_type float16


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))') 
python transcribe.py -i /home/peng/Documents/PROGRAM/GitHub/GPT-SoVITS/data/lain -o Data/lain/esd.list --speaker_name lain --compute_type float16

# python -c "import site;print(site.getsitepackages())"

python preprocess_all.py
```
```bash
# 上でつけたモデル名を入力。学習を途中からする場合はきちんとモデルが保存されているフォルダ名を入力。
model_name = "lain"
with open("default_config.yml", "r", encoding="utf-8") as f:
    yml_data = yaml.safe_load(f)
yml_data["model_name"] = model_name
with open("config.yml", "w", encoding="utf-8") as f:
    yaml.dump(yml_data, f, allow_unicode=True)



python train_ms_jp_extra.py --config Data/lain/config.json --model Data/lain --assets_root model_assets

python app.py --share --dir model_assets
```