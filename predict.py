import torch
from cog import BasePredictor, Input, Path
from diffusers import DiffusionPipeline

class Predictor(BasePredictor):
    def setup(self):
        """Carrega o modelo apenas uma vez (quando o container inicia)."""
        self.pipe = DiffusionPipeline.from_pretrained(
            "shuttleai/shuttle-3-diffusion",
            torch_dtype=torch.float16
        ).to("cuda")

    def predict(
        self,
        prompt: str = Input(description="Texto para gerar imagem"),
        width: int = Input(default=512, ge=256, le=2048),
        height: int = Input(default=512, ge=256, le=2048),
        steps: int = Input(default=20, ge=1, le=50),
        guidance: float = Input(default=7.5, ge=1, le=20),
    ) -> Path:
        """Gera a imagem a partir do prompt."""
        image = self.pipe(
            prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
        ).images[0]

        out_path = Path("output.png")
        image.save(out_path)
        return out_path
