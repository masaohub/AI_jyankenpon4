import av
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from PIL import Image
import time
import random
import pandas as pd
import lightgbm as lgb
import torch
from turn import get_ice_servers
from queue import Queue
import streamlit.components.v1 as stc
import base64

# https://github.com/whitphx/streamlit-webrtc


# データセットの読み込み
# GitHubリポジトリ内のファイル相対パス
file_relative_path = 'DataFrame.csv'

# データを読み込む
@st.cache_data
def load_data():
    data = pd.read_csv(file_relative_path)
    return data

# データを読み込む
data = load_data()

# 目標値（目的変数）の指定
target_column = 'target'  # 目標値の列名を適宜変更する必要があります
y = data[target_column]
x = data.drop(target_column, axis=1)  # 目標値を除いた特徴量をXとして使用

# 予測モデルの構築（LightGBM）
lgb_model = lgb.LGBMClassifier(
    boosting_type="gbdt",
    learning_rate=0.1,
    max_depth=3,
    num_leaves=5,
    n_estimators=200,
    objective="multiclass",
    random_state=42
)
lgb_model.fit(x, y)

yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_goochokipar.pt')  # 自分で学習したモデルのパスに変更してください。
# キューを作成
detected_classes_queue = Queue()

# グローバル変数として定義
detected_classes = None

def video_frame_callback(frame):
    global detected_classes  # グローバル変数を使用する宣言
    img = frame.to_ndarray(format="bgr24")

    # YOLOv5で画像から物体検出
    results = yolo_model(img)
    
    # 検出した物体の情報を描画
    rendered_img = results.render()[0]

    # 検出したクラスIDを取得
    detected_classes = results.pred[0][:, -1].cpu().numpy()

    # キューにクラスIDを追加
    detected_classes_queue.put(detected_classes)

    return av.VideoFrame.from_ndarray(rendered_img, format="bgr24")



# それぞれの手の画像を読み込む
image_path0 = 'images/ロボットちょき.png'
image0 = Image.open(image_path0) #COM用
image_path1 = 'images/ロボットぐー.png'
image1 = Image.open(image_path1) #COM用
image_path2 = 'images/ロボットぱー.png'
image2 = Image.open(image_path2) #COM用
image_path3 = 'images/ロボット.png'
image3 = Image.open(image_path3) #COM用
image_path4 = 'images/ロボット_勝ち.png'
image4 = Image.open(image_path4) #COM用
image_path5 = 'images/ロボット_負け.png'
image5 = Image.open(image_path5) #COM用

# 変数に代入する
choices = ['ちょき', 'ぐー　', 'ぱー　']
choices_mapping = {0: 'ちょき', 1: 'ぐー', 2: 'ぱー', 3:'-'}
# 対戦結果のグーチョキパーそれぞれの加算数値を 0 に初期化する
result0 = 0
result1 = 0
result2 = 0
win_lose = 0

if "n" not in st.session_state:
    st.session_state.n = 3
if "X1" not in st.session_state:
    st.session_state.X1 = 3
if "X2" not in st.session_state:
    st.session_state.X2 = 3
if "X3" not in st.session_state:
    st.session_state.X3 = 3

#　タイトルとテキストを記入
st.title('AI じゃん・けん・ぽん')

st.subheader('ボタンを押してゲームスタート')
clicked1 = st.button('スタート')  # ぐーボタン


# 選んだユーザーのカメラ、COMの手 を表示させる配置を設定'
user_col, com_col = st.columns(2)

with user_col:

    webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": get_ice_servers(),
            "iceTransportPolicy": "relay",
        },
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # COMの手の画像を初期表示
with com_col:
    if st.session_state.n == 0: #ちょき
        n_image = image0
    elif st.session_state.n == 1: #ぐー
        n_image = image1
    elif st.session_state.n == 2: #ぱー
        n_image = image2
    elif st.session_state.n == 3: #ノーマル
        n_image = image3
    elif st.session_state.n == 4: #ロボット_勝ち
        n_image = image4
    elif st.session_state.n == 5: #ロボット_負け
        n_image = image5


with st.sidebar:
    st.header('対戦回数を選んでね')
    number =st.number_input('対戦回数を選んでね', min_value=1, max_value=20,label_visibility="collapsed")
#    clicked1 = st.button('スタート')  # ぐーボタン

    st.header('対戦回数')
    # ボタンのクリック回数をカウントする変数
    # session_state は過去の数値を引き継いで加算する
    if "click_count" not in st.session_state:
        st.session_state.click_count = 0
    # スタートボタンをクリックすると回数を加算する
    if clicked1:
        st.session_state.click_count += 1
    # ボタンのクリック回数の表示
    st.write(f'{st.session_state.click_count} 回')

    st.header('対戦成績')
    # 対戦成績の回数をセット
    win_col, lose_col, eql_col = st.columns(3)
    if "win" not in st.session_state:
        st.session_state.win = 0
    if "lose" not in st.session_state:
        st.session_state.lose = 0
    if "eql" not in st.session_state:
        st.session_state.eql = 0

    # セッション状態の初期化とデータフレームの作成
    if "hand_log" not in st.session_state:
        df_new = pd.DataFrame(columns=['あなた', 'COM', '結果'])
        st.session_state.hand_log = df_new

    # データフレームで対戦ログを表示
    if st.session_state.click_count == 0: 
        st.write(st.session_state.hand_log)

# COMが出す手をランダムで決める
if  0 < st.session_state.click_count < 4:
    st.session_state.n = random.randint(0, 2)
    print('random')
    print('COM', st.session_state.n)

def COM_hand(placeholder):
    # COMの手の画像を確定
    if st.session_state.n == 0:
        n_image = image0
    elif st.session_state.n == 1:
        n_image = image1
    elif st.session_state.n == 2:
        n_image = image2
    elif st.session_state.n == 3: #ノーマル
        n_image = image3
    elif st.session_state.n == 4: #ロボット_勝ち
        n_image = image4
    elif st.session_state.n == 5: #ロボット_負け
        n_image = image5
    placeholder.image(n_image)
        
# ユーザーの手のクラス名を表示
st.session_state.X3 = st.session_state.X2
st.session_state.X2 = st.session_state.X1


def play_audio(audio_path):
    #じゃんけんぽんの音声を流す
    audio_placeholder = st.empty() #入力する音声ファイル
    with open(audio_path, "rb") as file_:
        contents = file_.read()
        
    audio_str = "data:audio/wav;base64,%s" % (base64.b64encode(contents).decode())
    audio_html = """
                    <audio autoplay=True>
                    <source src="%s" type="audio/wav" autoplay=True>
                    Your browser does not support the audio element.
                    </audio>
                """ % audio_str
    audio_placeholder.empty()
    time.sleep(0.5)  #これがないと上手く再生されません
    audio_placeholder.markdown(audio_html, unsafe_allow_html=True)

# 手のボタン をクリックすると、イベント開始
if clicked1:

    audio_path1 = 'sounds/じゃんけんぽん.wav'
    play_audio(audio_path1)

    # じゃん・けん・ぽん を順番に表示
    jyan_col1, ken_col2, pon_col3 = st.columns(3)
    with jyan_col1:
        time.sleep(0.5)
        st.subheader('じゃん')
    with ken_col2:
        time.sleep(0.8)
        st.subheader('けん')
    with pon_col3:
        time.sleep(0.8)
        st.subheader('ぽん')

    time.sleep(1)

    # 物体が検出されるまでループ
    while True:
        if detected_classes is not None and len(detected_classes) > 0:
            first_class = int(detected_classes[0])
            st.session_state.X1 = first_class
            print('検出した手')
            print(first_class, choices_mapping[st.session_state['X1']])
            break
        time.sleep(0.1)

    # ユーザーの手のクラス名を表示
#    st.session_state.X3 = st.session_state.X2
#    st.session_state.X2 = st.session_state.X1


# COMの手の画像を表示
with com_col:
    placeholder = st.empty()
    COM_hand(placeholder) 

if clicked1:
    def janken_result(detected_classes):
        result0 = 0
        result1 = 0
        result2 = 0
        win_lose = ""

        if detected_classes is not None:
            if detected_classes[0] == st.session_state.n:  # あいこ
                st.header('あいこ')
                result2 = 1
                win_lose = "△"
            elif (detected_classes[0] - st.session_state.n + 3) % 3 == 1:  # かち
                st.header('あなたの かち')
                result0 = 1
                win_lose = "〇"
            else:  # まけ
                st.header('あなたの まけ')
                result1 = 1
                win_lose = "×"

        st.session_state.win += result0
        st.session_state.lose += result1
        st.session_state.eql += result2

        choices_mapping = {0: 'ちょき', 1: 'ぐー', 2: 'ぱー'}  # 例として
        df_new = pd.DataFrame({'あなた': [choices_mapping[detected_classes[0]]],
                                'COM': [choices_mapping[st.session_state.n]],
                                '結果': [win_lose]})
        st.session_state.hand_log = pd.concat([st.session_state.hand_log, df_new], axis=0).reset_index(drop=True)
        st.sidebar.dataframe(st.session_state.hand_log)

        # Streamlitのセッション変数の初期設定（必要な場合）
        if 'win' not in st.session_state:
            st.session_state.win = 0
        if 'lose' not in st.session_state:
            st.session_state.lose = 0
        if 'eql' not in st.session_state:
            st.session_state.eql = 0
        if 'hand_log' not in st.session_state:
            st.session_state.hand_log = pd.DataFrame(columns=['あなた', 'COM', '結果'])

    # 関数を呼び出す    
    janken_result(detected_classes)



    log_col1, pred_col2 = st.columns(2)
    with log_col1:
            st.write('1回前:', choices_mapping[st.session_state['X1']])
            st.write('2回前:', choices_mapping[st.session_state['X2']])
            st.write('3回前:', choices_mapping[st.session_state['X3']])

    with pred_col2:
        # データフレームを作成
        X_Value = pd.DataFrame([[st.session_state.X3, st.session_state.X2, st.session_state.X1]], columns=['X3', 'X2', 'X1'])
        # 予測値のデータフレーム
        st.session_state.n = lgb_model.predict(X_Value)
        print(X_Value)
        st.session_state.n = st.session_state.n.item()
        print('学習結果COM次の手:',choices_mapping[st.session_state.n])
        if st.session_state.click_count > 2:
            com_next_hand = random.randint(1, 3)
            print('random 挑発手:',com_next_hand)
            if com_next_hand == 1:
                com_choice_hand = random.randint(0, 2)
                next_col1, hand_col2, maybe_col3 = st.columns(3)
                with next_col1:
                    st.write('つぎは ')
                with hand_col2:
                    st.subheader(choices_mapping[com_choice_hand])                
                with maybe_col3:
                    st.write(' をだそうかな')

with win_col:
        st.write(f'勝ち {st.session_state.win} 回')
with lose_col:
        st.write(f'負け {st.session_state.lose} 回')
with eql_col:
        st.write(f'あいこ {st.session_state.eql} 回')


if st.session_state.click_count == number:
    if st.session_state.win > st.session_state.lose:
        st.title('お・・おまえのかちだ!!!')
        audio_path2 = 'sounds/お前の勝ち.wav'
        st.balloons()
        st.session_state.n = 5
    elif st.session_state.win < st.session_state.lose:
        st.title('ＡＩはつよいんだぞ')
        audio_path2 = 'sounds/AIは強い.wav'
        st.session_state.n = 4
    else:
        st.title('なかなかやるじゃないか')
        audio_path2 = 'sounds/なかなかやるな.wav'
        st.session_state.n = 3
    play_audio(audio_path2)
    st.session_state.click_count = 0
    # COMの手の画像を表示
    with com_col:
        time.sleep(3)
        COM_hand(placeholder) 


if st.button("もういちどあそぶ"):
    st.session_state.click_count = 0
    st.session_state.win = 0
    st.session_state.lose = 0
    st.session_state.eql = 0
    st.session_state.hand_log = ''
    df_new = pd.DataFrame(columns=['あなた', 'COM', '結果'])
    st.session_state.hand_log = df_new
    st.experimental_rerun()

