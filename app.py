import argparse
from pathlib import Path

import gradio as gr
import torch
from fastapi import APIRouter, FastAPI
from fastapi.openapi.docs import get_swagger_ui_html

from config import get_path_config
from gradio_tabs.dataset import create_dataset_app
from gradio_tabs.inference import create_inference_app
from gradio_tabs.merge import create_merge_app
from gradio_tabs.style_vectors import create_style_vectors_app
from gradio_tabs.train import create_train_app
from server_fastapi import add_server_api
from style_bert_vits2.constants import GRADIO_THEME, VERSION
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.tts_model import TTSModelHolder

# このプロセスからはワーカーを起動して辞書を使いたいので、ここで初期化
pyopenjtalk_worker.initialize_worker()

# dict_data/ 以下の辞書データを pyopenjtalk に適用
update_dict()

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=None)
parser.add_argument("--no_autolaunch", action="store_true")
parser.add_argument("--share", action="store_true")
# parser.add_argument("--skip_default_models", action="store_true")
args = parser.parse_args()
device = args.device
if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"

# if not args.skip_default_models:
#     download_default_models()

path_config = get_path_config()
model_holder = TTSModelHolder(Path(path_config.assets_root), device)

with gr.Blocks(theme=GRADIO_THEME) as app:
    gr.Markdown(f"# Style-Bert-VITS2 WebUI (version {VERSION})")
    with gr.Tabs():
        with gr.Tab("音声合成"):
            create_inference_app(model_holder=model_holder)
        with gr.Tab("データセット作成"):
            create_dataset_app()
        with gr.Tab("学習"):
            create_train_app()
        with gr.Tab("スタイル作成"):
            create_style_vectors_app()
        with gr.Tab("マージ"):
            create_merge_app(model_holder=model_holder)

    api_app, local_url, _ = app.launch(
        inbrowser=not args.no_autolaunch,
        share=args.share,
        server_name=args.host,
        prevent_thread_lock=True,
        server_port=args.port,
    )
    router = APIRouter()

    add_server_api(router, device)
    api_app.include_router(router, prefix="/extern_api")


    def add_docs_api(app: FastAPI):
        temp_api = FastAPI()
        add_server_api(temp_api)

        @app.get("/extern_api/openapi.json", include_in_schema=False)
        async def _():
            return temp_api.openapi()

        @app.get("/extern_api/docs", include_in_schema=False)
        async def _():
            return get_swagger_ui_html(openapi_url="/extern_api/openapi.json", title="API documentation")


    add_docs_api(api_app)

    print("API server started.", local_url)
    app.block_thread()
