import os
from playsound import playsound
from modules.fish_speech.tools.llama.generate import main as generate
from pathlib import Path
from modules.fish_speech.tools.vqgan.inference import main as infer

class VoiceService:
    def __init__(self):
        self._output_dir = "outputs/"
        os.makedirs(self._output_dir, exist_ok=True)

    def fishspeech(self, text,character="Aeliana"):
        # for the first time must generate prompt_token from source voice
        # infer(input_path=Path("source_voice/sango.mp3"), output_path=Path(self._output_dir+"fake5.wav"),
        #        checkpoint_path="checkpoints/fish-speech-1.2/firefly-gan-vq-fsq-4x1024-42hz-generator.pth",
        #        config_name="firefly_gan_vq", device="cuda")
        if character=="Aeliana":
            prompt_text="lama - let us celebrate our diversity as it adds color to life itself and creates a world that is more beautiful than ever before – lama! So go ahead; don't be afraid of being yourself or loving whomever you wish. After all, love knows no boundaries in this vast universe—lama - let's continue celebrating our diversity together with open hearts and minds- Lama!!! Let us take pride where we are today while working towards an even more inclusive tomorrow – lama!"
            n=0
            

        if character=="Elara":
            prompt_text="You seem like you're in the mood for a fun conversation today! So, what shall we dive into? We could chat about anything you like—whether it's something technical, exciting ideas you're working on, or maybe even something casual. I'm all ears and ready to keep things engaging. Just let me know what's on your mind, and we'll go from there!"
            n=3
            
        if character=="Lyra":
            prompt_text="You seem like you're in the mood for a fun conversation today! So, what shall we dive into? We could chat about anything you like—whether it's something technical, exciting ideas you're working on, or maybe even something casual. I'm all ears and ready to keep things engaging. Just let me know what's on your mind, and we'll go from there! Ara Ara."
            n=5
        
        generate(text=text,
                 prompt_text=[prompt_text],#for xianyun You seem like you're in the mood for a fun conversation today! So, what shall we dive into? We could chat about anything you like—whether it's something technical, exciting ideas you're working on, or maybe even something casual. I'm all ears and ready to keep things engaging. Just let me know what's on your mind, and we'll go from there! Ara Ara for sango
                 prompt_tokens=[Path(self._output_dir+f"fake{n}.npy")], #fake3.npy is xianyun, fake0.npy is bailu, fake5.npy is songo
                 checkpoint_path=Path("./checkpoints/fish-speech-1.2"),
                 half=True,
                 device="cuda",
                 num_samples=1,
                 max_new_tokens=1000,
                 top_p=0.7,
                 repetition_penalty=1.3,
                 temperature=0.4,
                 compile=False,
                 seed=42,
                 iterative_prompt=True,
                 chunk_length=500)

        infer(input_path=Path("codes_0.npy"), output_path=Path(self._output_dir+"output0.wav"),
              checkpoint_path="./checkpoints/fish-speech-1.2/firefly-gan-vq-fsq-4x1024-42hz-generator.pth",
              config_name="firefly_gan_vq", device="cuda")

        #self.play(self._output_dir+"output0.wav")

    def play(self, temp_audio_file):
        playsound(temp_audio_file)

if __name__=="__main__":
    vs=VoiceService()
    vs.fishspeech("lama you are gay nigga",character="Aeliana")
    vs.play("./outputs/output0.wav")