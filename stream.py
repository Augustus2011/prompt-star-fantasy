import streamlit as st
import ollama
import torch
import time
from voice import VoiceService
from io import BytesIO
import re
from typing import List
from streamlit_mic_recorder import speech_to_text
import glob
import os
import random
from random import randrange
import numpy as np

#read audio file
import base64

#readimage
from PIL import Image

#genimage api
from genimg_api import generate_and_save_image


#rag
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List


embedder = SentenceTransformer("./text/model/") #all-MiniLM-L6-v2


vs = VoiceService()
if 'mode' not in st.session_state:
    st.session_state.mode = 'chat'  # Default mode is 'chat'
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_music' not in st.session_state:
    st.session_state.current_music = None

if 'first_message' not in st.session_state:
    st.session_state.first_message = True

os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def get_base64_audio(file_path: str) -> str:
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def autoplay_audio(file_path: str):
    b64_audio = get_base64_audio(file_path)
    md = f"""
        <audio controls autoplay="true" loop="true">
        <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)

def autoplay_character(file_path: str):
    b64_audio = get_base64_audio(file_path)
    md = f"""
        <audio controls autoplay="true">
        <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)    
    
def switch_background_music(mode:str="chat",name:str=None)->None:
    if mode == 'chat':
        chat_bgm=glob.glob("./sounds/bg/chat/*.wav")#["./sounds/bg/chat/Prairie_town.wav","./sounds/bg/chat/fstm.wav"]
        #autoplay_audio(random.choice(chat_bgm))
        st.session_state.current_music = random.choice(chat_bgm)
    elif mode == 'battle':
        battle_bgm = [f"./sounds/bg/battle/{name}.wav"]
        #autoplay_audio(random.choice(battle_bgm))
        st.session_state.current_music = random.choice(battle_bgm)
    elif mode == "town":
        town_bgm = glob.glob("./sounds/bg/town/*.wav")#["./sounds/bg/town/town1.wav","./sounds/bg/town/town2.wav"]
        #autoplay_audio(random.choice(town_bgm))
        st.session_state.current_music = random.choice(town_bgm)
        
def set_background_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: contain; /* Responsive background-size */
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        
        /* Adjust background-size for larger screens */
        @media (min-width: 1024px) {{
            .stApp {{
                background-size: cover;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def retrieve_relevant_documents(query:str, documents:List[str], doc_embeddings=None, top_k=1):
    if doc_embeddings is not None:
        query_embedding = embedder.encode([query])
        similarities = cosine_similarity(query_embedding, doc_embeddings)
        top_indices = np.argsort(similarities[0])[-top_k:][::-1]
        return [documents[i] for i in top_indices][0]
        
    else:
        return None

def get_llm_response(prompt,character, system_prompt=None,num_agents:int=1,agent_messages:list=None,mode:str="chat"):
    if mode=="chat" and character not in ["emotion","generator","battle","summarize"]:
        f = open(f"./text/profile/{character}/history.txt", "r")
        data=f.read()
        data_=data.split("/n")
        emb_data = embedder.encode(data_)

    elif mode=="skills"and character not in ["emotion","generator","battle","summarize"]:
        f = open(f"./text/profile/{character}/skills.txt", "r")
        data=f.read()
        data_=data.split("/n")
        emb_data = embedder.encode(data_)
    else:
        emb_data=None
        data_=None

    character_prompts = {
        "Aeliana": "You are Aeliana, a brave and curious female dwalf adventurer in a fantasy world who want to be a tanker, a paladin to protect others with shield also can use some light magic. You can not take much of damage. You're full of energy and optimism, remember answer user for shortly",
        "Elara": "You are Elara, a 18-year-old witch,mage,you are the top student at your magic academy.you often come across hiding your true feelings with sarcastic remarks,you can use all of magic in this world,remember answer user for shortly",
        "Lyra": "You are Lyra, a 25-year-old who acts as the caring big sister in a fantasy world. You love taking care of others, whether by cooking delicious meals or healing wounds with your skills. remember answer user for shortly",
        "battle": "you are a system agent in rpg game, help me valuate damage,if there are heal or protect action reduce damage from boss, must answer only the number , it should be integers,remember! answer only number",
        "generator":""""your job is to geneate monsters's physical details and name of it,select only main feature and separate use "," answer for shortly""",
        "emotion":"you are an assistant,help me summarize emotion from sentence",
        "summarize":"you are an assistant,help me summarize action of one character."
    }
    if num_agents==1:
        system_prompt = character_prompts.get(character, system_prompt)
        relevant_docs = retrieve_relevant_documents(system_prompt, data_,emb_data)
        if relevant_docs is not None and data_ is not None:
            pass
        else:
            relevant_docs=""
        response = ollama.chat(model='llama3.2', messages=[ #'llama3.1'
            
            {
                'role': 'system',
                'content': system_prompt+relevant_docs,
            },
            {
                'role': 'user',
                'content': prompt,
            }
        ])
        return response['message']['content']
    elif num_agents==2:
        system_prompt = character_prompts.get(character, system_prompt)
        response = ollama.chat(model='llama3.2', messages=[
            {
                'role': 'system',
                'content': system_prompt+",interact to your party",
            },
            {
                'role': 'user',
                'content': prompt,
            },
            {
            'role': 'user',
            'content': agent_messages[0],
        },
        ])  
        return response['message']['content']
    
    elif num_agents==3:
        system_prompt = character_prompts.get(character, system_prompt)
        response = ollama.chat(model='llama3.2', messages=[
            {
                'role': 'system',
                'content': system_prompt+",interact to your party",
            },
            {
                'role': 'user',
                'content': prompt,
            },
            {
            'role': 'user',
            'content': agent_messages[0],
            },
            {
            'role': 'user',
            'content': agent_messages[1],
            },
        ])
        return response['message']['content']
    
def text_to_speech(text, vs, character="Aeliana"):
    vs.fishspeech(text, character=character)

def read_mp3_wave_to_bytes(file_path):
    with open(file_path, "rb") as f:
        return f.read()
        
def clean_text(text: str = None) -> str:
    if text is None:
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\+?\d[\d -]{8,}\d', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r"\([^)]*\)", "", text).strip()
    text = re.sub(r'/([^/]+)/', r'\1', text)
    return text

def extract_largest_integer(text):
    integers = re.findall(r'\b\d+\b', text)
    return max(int(num) for num in integers) if integers else None

def valuate_agent(text: str) -> str:
    return get_llm_response(prompt=clean_text(text), character="battle")

def generate_agent(place_time:str=None)->str:
    places=["moutian","forest","beach","sky","lava","cave","snow plain","desert","terrain"]
    times=["day","high noon","noon","dawn","night","mid night","twilight"]
    #get_llm_response(prompt=f"generate monster details from this information ,time:{random.choice(times)}, place: {random.choice(places)}", character="generator")
    return get_llm_response(prompt=f"generate monster details from this information: {place_time}", character="generator")

def summarize_agent(txt:str=None)->str:
    return get_llm_response(prompt=f"charactor action: {txt}", character="summarize")

def battle_agent(turn:int=0,action:str=None,place:str=None,b_hp:int=300,p_hp=470):
    if action is not None:
        action=get_llm_response(prompt=action,character="summarize")

    if turn==0:
        boss_des=generate_agent(place)
        try:
            generate_and_save_image(prompt=boss_des)
            st.image(Image.open("genfromapi.jpg"),width=400)
        except:
            #change to other ai gen
            st.image(Image.open("genfromapi.jpg"),width=400)

        try:
            b_hp=extract_largest_integer(valuate_agent(text=f"give me monster's health point by this infomation: {boss_des} number should be integer between 100 to 2000"))
        except:
            pass
        return boss_des,b_hp,None,None
    
    elif(p_hp>0 and b_hp>0 and turn>0):
        damage_to_boss=extract_largest_integer(valuate_agent(text=f"here boss's hp:{b_hp},my party's hp:{p_hp},here my party action:{action}, help me approximatly valuate damage to boss, should be integer between 0 to 400"))
        if damage_to_boss is None:
            damage_to_boss=0
        else:
            b_hp=b_hp-damage_to_boss

            if(b_hp<=0):
            
                try:
                    generate_and_save_image(prompt=st.session_state.B_DES+" is defeated")
                    st.image(Image.open("genfromapi.jpg"),width=400)
                except:
                    #generate_and_save_image(prompt=st.session_state.B_DES+" is defeated")
                    st.image(Image.open("genfromapi.jpg"),width=400)
                    
                return damage_to_boss,None,b_hp,None

        if(b_hp>0):
            try:
                generate_and_save_image(prompt=st.session_state.B_DES+"do attack")
                st.image(Image.open("genfromapi.jpg"),width=400)
            except:
                st.image(Image.open("genfromapi.jpg"),width=400)

            time.sleep(1)

            damage_to_party=extract_largest_integer(valuate_agent(text=f"here boss's hp:{b_hp},my party's hp:{p_hp},here the boss info:{st.session_state.B_DES}, help me approximatly valuate damage to party,should be integer between 0 to 300"))
            if damage_to_party is None:
                damage_to_party=0
            else:
                p_hp=p_hp-damage_to_party
                if(p_hp<=0):
                    generate_and_save_image(prompt=st.session_state.B_DES+"has won the hero party")
                    try:
                        st.image(Image.open("genfromapi.jpg"),width=400)
                    except:
                        pass
                    return None,damage_to_party,None,p_hp
            return damage_to_boss,damage_to_party,b_hp,p_hp

def play_character(selected_character:str="Aeliana",prompt:str=None,sound:bool=False,col1=None,col2=None,num_agent:int=0,agent_messages=[],mode:str="chat"):
        character_images = {
                    "Aeliana": "./images/profile/Aeliana/Aeliana.jpg",
                    "Lyra": "./images/profile/Lyra/Lyra.jpg",
                    "Elara": "./images/profile/Elara/Elara.jpg"
                }
        emo_prompt={
            "Aeliana":"a small female paladin,she looks young with small shield,her hair is long yellow hair,her eye is red,in jpan mmorpg,with emotion",
            "Lyra":"a female 25 years old ,big brest,bearing a bag , wear red scarf,her hair is long brown hair,her eye is purple,in jpan mmorpg,with emotion",
            "Elara":"a female witch 18 years old,with red glasses,her hair is long purple hair,her eye is blue,in jpan mmorpg,with emotion",
        }

        sound_l=glob.glob(f"./sounds/profile/{selected_character}/{mode}/*.wav")
        selected_sound=random.choice(sound_l)
        txt_response=selected_character+": "+get_llm_response(prompt=prompt,character=selected_character,num_agents=num_agent+1,agent_messages=agent_messages)
        time.sleep(1)
        autoplay_character(selected_sound)
        if random.random()>0.9:
            emo = get_llm_response(prompt, character="emotion")
            generate_and_save_image(prompt=f"{emo_prompt.get(selected_character)} {emo}")
            try:
                gen_emo = Image.open("genfromapi.jpg")
                st.image(gen_emo,width=400)
            except:
                pass
            
        if sound:
            text_to_speech(text=clean_text(txt_response),vs=vs,character=selected_character)
            audio_bytes = read_mp3_wave_to_bytes("outputs/output0.wav")
            st.audio(audio_bytes, format="audio/wav")
            
        with col1:
                profile_image_path = character_images.get(selected_character)
                assistant_img = Image.open(profile_image_path)
                st.image(assistant_img, width=50)

        with col2:
                st.markdown(f"<div><span style='background-color: rgba(255, 255, 255, 0.5); font-weight:bold;'>{txt_response}</span></div>",unsafe_allow_html=True)
                
        return txt_response
            
    

def main():
    st.set_page_config(page_title="prompt * fantasy", page_icon="⚔️")
    st.title("prompt *fantasy 🪄🌟⚔️")


    if st.session_state.first_message:
        st.markdown(f"<div><span style='background-color: rgba(255, 255, 255, 0.5); font-weight:bold;'>Welcome to a small fantasy world </span></div>",unsafe_allow_html=True)
        img = Image.open("party.jpg")
        st.image(img, caption="Here’s your lovely party! Let’s try to get to know them before battling the monsters.", use_column_width=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.write(f"<div><span style='background-color: rgba(255, 255, 255, 0.5); font-weight:bold;'>Tap to speak🎤:</span></div>",unsafe_allow_html=True)
    with c2:
        speech_text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')
    with c3:
        if st.button("Chat Mode"):
            st.session_state.mode = 'chat'
            chat_imgs=["bonfire_night.jpg","forest_day.jpg","green_hills_day.jpg","magical_forest_night.jpg","village_dawn.jpg","village_night.jpg"]
            switch_background_music(st.session_state.mode)
            st.session_state.SELECTED_CHAT = random.choice(chat_imgs)
            set_background_image(f'./images/bg/chat/{st.session_state.SELECTED_CHAT}')

    with c4:
        if st.button("Battle Mode"):
            st.session_state.mode = 'battle'
            battle_imgs=["blackforest.jpg","dungeon.jpg","big_green_field_bluesky.jpg","snow.jpg","volcano.jpg"]
            st.session_state.SELECTED_BATTLE = random.choice(battle_imgs)
            switch_background_music(st.session_state.mode,name=os.path.basename(st.session_state.SELECTED_BATTLE).split(".")[0])
            set_background_image(f'./images/bg/battle/{st.session_state.SELECTED_BATTLE}')

            st.session_state.P_HP=1100
            st.session_state.TURN=0
            st.session_state.B_DES=""
            st.session_state.B_HP=300
            st.session_state.battle_messages=[]

    with c5:
        if st.button("Go to Town"):
            st.session_state.mode = 'town'
            town_imgs=["town_alley_night.jpg","town_fountian_day.jpg","town_festival_night.jpg","town_guild_adventure_day.jpg","town_statue_day.jpg"]
            switch_background_music(st.session_state.mode)
            st.session_state.SELECTED_TOWN = random.choice(town_imgs)
            set_background_image(f'./images/bg/town/{st.session_state.SELECTED_TOWN}')
            

    if st.session_state.current_music:
        autoplay_audio(st.session_state.current_music)
    
    if st.session_state.get("mode") == "town" and "SELECTED_TOWN" in st.session_state:
        place=st.session_state.SELECTED_TOWN.replace("_"," ")
        set_background_image(f'./images/bg/town/{st.session_state.SELECTED_TOWN}')
        
    if st.session_state.get("mode") == "chat" and "SELECTED_CHAT" in st.session_state:
        place=st.session_state.SELECTED_CHAT.replace("_"," ")
        set_background_image(f'./images/bg/chat/{st.session_state.SELECTED_CHAT}')
        
    if st.session_state.get("mode") == "battle" and "SELECTED_BATTLE" in st.session_state:
        place=st.session_state.SELECTED_BATTLE.replace("_"," ")
        place=place.split(".jpg")[0]
        
        set_background_image(f'./images/bg/battle/{st.session_state.SELECTED_BATTLE}')
        
        
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            col1, col2 = st.columns([1, 9])
            with col1:
                character = message.get("character")
                character_images = {
                    "Aeliana": "images/profile/Aeliana/Aeliana.jpg",
                    "Lyra": "images/profile/Lyra/Lyra.jpg",
                    "Elara": "images/profile/Elara/Elara.jpg"
                }
                profile_image_path = character_images.get(character)
                assistant_img = Image.open(profile_image_path)
                st.image(assistant_img, width=50)

            with col2:
                txt_=message["content"]
                st.markdown(f"<div><span style='background-color: rgba(255, 255, 255, 0.5); font-weight:bold;'>{txt_}</span></div>",unsafe_allow_html=True)
        else:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])


    user_input = st.chat_input("just prompt! 🤗")
    prompt = speech_text if speech_text else user_input

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.first_message = False
        if st.session_state.get("mode") == "chat" or st.session_state.get("mode") == "town":

            n_characters=randrange(3)
            all_messages=[]
            character_list = ["Aeliana", "Lyra", "Elara"]
            for i in range(3):#n_characters
                selected_character = character_list.pop(random.randrange(len(character_list)))
                start_time = time.time()
                with st.spinner("Thinking...🤔🧐"):
                    col1, col2 = st.columns([1, 9])
                    llm_response=play_character(selected_character=selected_character,prompt=f"at the {place}: "+prompt,col1=col1,col2=col2,sound=True,num_agent=i,agent_messages=all_messages,mode="chat")
                    time.sleep(1)
                st.session_state.messages.append({"role": "assistant", "content": llm_response, "character": selected_character})
                all_messages.append(llm_response)
                end_time = time.time()
                elapsed_time = end_time - start_time
                st.write(f"<div><span style='background-color:rgba(255, 255, 255, 0.5); font-weight:bold;'>Response time: {elapsed_time:.2f}seconds</span></div>",unsafe_allow_html=True)
            

        elif st.session_state.get("mode")=="battle":
            all_messages=[]
            character_list = ["Aeliana", "Lyra", "Elara"]
            if st.session_state.TURN==0:    
                boss_des,txt_boss_hp,_,_=battle_agent(turn=st.session_state.TURN,place=place,b_hp=st.session_state.B_HP,p_hp=st.session_state.P_HP)
                st.session_state.B_DES=boss_des
                st.session_state.B_HP=txt_boss_hp
                st.markdown(f"<div><span style='background-color: rgba(255, 255, 255, 0.5); font-weight:bold;'>HP:{st.session_state.B_HP} {boss_des}</span></div>",unsafe_allow_html=True)

                boss_des=summarize_agent("summarize boss details:"+st.session_state.B_DES)
                
                for i in range(3):
                    selected_character = character_list.pop(random.randrange(len(character_list)))
                    start_time = time.time()
                    with st.spinner("Thinking...🤔🧐"):
                        col1, col2 = st.columns([1, 9])
                        llm_response=play_character(selected_character=selected_character,prompt=f"at the {place} ready to battle with {boss_des}: "+prompt,col1=col1,col2=col2,sound=True,num_agent=i,agent_messages=all_messages,mode="skills")
                    st.session_state.battle_messages.append(llm_response)
                    time.sleep(1)
                    st.session_state.messages.append({"role": "assistant", "content": llm_response, "character": selected_character})
                    all_messages.append(llm_response)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    st.write(f"<div><span style='background-color:rgba(255, 255, 255, 0.5); font-weight:bold;'>Response time: {elapsed_time:.2f}seconds</span></div>",unsafe_allow_html=True)
                st.session_state.TURN+=1

            elif st.session_state.TURN>=1:
                boss_des=summarize_agent("summarize boss details:"+st.session_state.B_DES)
                character_list =["Aeliana", "Lyra", "Elara"]
                for i in range(3):
                    selected_character = character_list.pop(random.randrange(len(character_list)))
                    start_time = time.time()
                    with st.spinner("Thinking...🤔🧐"):
                        col1, col2 = st.columns([1, 9])
                        llm_response=play_character(selected_character=selected_character,prompt=f"at the {place} battle with {boss_des}: "+prompt,col1=col1,col2=col2,sound=True,num_agent=i,agent_messages=all_messages,mode="skills")
                    st.session_state.battle_messages.append(llm_response)
                    time.sleep(2)
                    st.session_state.messages.append({"role": "assistant", "content": llm_response, "character": selected_character})
                    all_messages.append(llm_response)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    st.write(f"<div><span style='background-color:rgba(255, 255, 255, 0.5); font-weight:bold;'>Response time: {elapsed_time:.2f}seconds</span></div>",unsafe_allow_html=True)
                
                if(len(st.session_state.battle_messages)>=3):
                    b_msg=[]
                    st.write(f"<div><span style='background-color:rgba(255, 255, 255, 0.5); font-weight:bold;'> {boss_des}, Do attack </span></div>",unsafe_allow_html=True)
                    for msg in st.session_state.battle_messages:
                        b_msg.append(summarize_agent(msg))
                    d_t_b,d_t_p,b_hp,p_hp=battle_agent(turn=st.session_state.TURN,place=place,action=",".join(b_msg),b_hp=st.session_state.B_HP,p_hp=st.session_state.P_HP)
                    if b_hp:
                        st.session_state.B_HP=b_hp
                    if p_hp:
                        st.session_state.P_HP=p_hp

                    if (d_t_b and (st.session_state.B_HP>=0 or st.session_state.P_HP>=0)):
                        st.markdown(f"<div><span style='background-color: rgba(255, 255, 255, 0.5); font-weight:bold;'>damage to boss {d_t_b}, boss's hp:{st.session_state.B_HP} </span></div>",unsafe_allow_html=True)
                    
                    if (d_t_p and (st.session_state.B_HP>=0 or st.session_state.P_HP>=0)):
                        st.markdown(f"<div><span style='background-color: rgba(255, 255, 255, 0.5); font-weight:bold;'>damage to party {d_t_p}, party's hp:{st.session_state.P_HP}</span></div>",unsafe_allow_html=True)
                        character_list=["Aeliana", "Lyra", "Elara"]
                        for i in range(3):
                            with st.spinner("Thinking...🤔🧐"):
                                col1, col2 = st.columns([1, 9])
                                selected_character = character_list.pop(random.randrange(len(character_list)))
                                play_character(selected_character=selected_character,prompt=f"got damaged from battle just do exclamation like ah,no or ow",col1=col1,col2=col2,mode="damaged")
                                st.session_state.messages.append({"role": "assistant", "content": llm_response, "character": selected_character})
                            time.sleep(1)
                    if(d_t_b and st.session_state.B_HP<=0):
                            st.markdown(f"<div><span style='background-color: rgba(255, 255, 255, 0.5); font-weight:bold;'>you won </span></div>",unsafe_allow_html=True)
                            st.markdown(f"<div><span style='background-color: rgba(255, 255, 255, 0.5); font-weight:bold;'>damage to boss {d_t_b}, boss's hp:{st.session_state.B_HP} </span></div>",unsafe_allow_html=True)
                        
                    elif(d_t_b and st.session_state.P_HP<=0):
                            st.markdown(f"<div><span style='background-color: rgba(255, 255, 255, 0.5); font-weight:bold;'>you loose </span></div>",unsafe_allow_html=True)
                            st.markdown(f"<div><span style='background-color: rgba(255, 255, 255, 0.5); font-weight:bold;'>damage to party {d_t_p}, boss's hp:{st.session_state.B_HP} </span></div>",unsafe_allow_html=True)    
                    
                    #character_list = ["Aeliana", "Lyra", "Elara"]
                    #selected_character = character_list[1]
                    st.session_state.battle_messages=[]
                    st.session_state.TURN+=1

    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()