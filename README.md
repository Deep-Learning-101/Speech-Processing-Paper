---
layout: default
title: 2026 語音處理資源懶人包 (Speech AI) | ASR, TTS & 聲紋辨識 | Deep Learning 101
description: 2026 最新開源語音處理 (Speech AI) 資源與模型比較。涵蓋免切片語音辨識 (ASR)、極速人聲克隆 (TTS)、語音增強去噪等技術，收錄 Whisper 魔改版與 VibeVoice 等企業級落地解決方案。
permalink: /Speech-Processing
lang: zh-Hant
schema_type: service
service_type: AI Consulting
---

{% include header.html %}

# 🎤 語音處理 (Speech)・必讀資源總整理

> **編者按：** 本頁面彙整了語音處理領域的前沿技術。包含自動語音辨識、語音合成、語者識別與語音轉換的經典論文與開源工具。
>
> 如果您想尋找更詳細的筆記，歡迎訪問 **GitHub Repository**：
> 👉 [**GitHub: Speech-Processing-Paper**](https://github.com/Deep-Learning-101/Speech-Processing-Paper) (歡迎 Star ⭐)

---

{% include ai-share.html %}

---

# 語音處理 (Speech Processing)

## 🎙️ 語音識別 / 合成平台價格比較
~2025/04

| 名稱 | 功能 | 網址 | 說明 |
|------|------|------|------|
| [Whisper (開源)](https://github.com/openai/whisper) | 語音識別、翻譯 | 每分鐘150字 × 10分鐘 = 1500字 |
| [Fish Audio](https://speech.fish.audio/zh/) | 語音識別、語音合成 | TTS：英文 $0.0225，中文 $0.0675；ASR：30分鐘 = $0.18 |
| [Deepgram](https://deepgram.com/pricing) | 語音識別 | TTS：英文 $0.02025，中文 $0.06075；ASR：30分鐘 = $0.147 |
| [Microsoft Azure](https://azure.microsoft.com/zh-tw/pricing/details/cognitive-services/speech-services/) | 語音合成 | TTS：英文 $0.036，中文 $0.108；ASR：即時轉錄 $1/小時，超額 $0.8/小時 |
| [Amazon Polly](https://aws.amazon.com/tw/polly/pricing) | 語音合成 | TTS：英文 $0.024，中文 $0.072 |
| [Google WaveNet](https://cloud.google.com/text-to-speech/pricing) | 語音合成 | TTS：英文 $0.024，中文 $0.072 |
| [Google Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/pricing?hl=zh-tw#gemini-models) | 大型語言模型 | Gemini/Claude 定價頁 |
| [Google Cloud VM](https://cloud.google.com/compute/vm-instance-pricing?hl=zh-tw#sharedcore_machine_types) | 虛擬機器 | VM 執行個體定價頁面 |

---

### **文章目錄**
- [Speech Processing (語音處理)](#speech-processing)
- [Speech Recognition (語音識別)](#speech-recognition)
- [Speaker Recognition (語者識別)](#speaker-recognition)
- [Speech Enhancement (語音增強)](#speech-enhancement)
- [Speaker Separation (語者分離)](#speaker-separation)
- [Speech Synthesis (語音合成)](#speech-synthesis)
- [Speech Datasets (開源語音資料)](#speech-datasets)
- [Speech-Applications(語音處理綜合應用)](#speech-applications)


## Speech-Processing
**🗣️ Speech Processing (語音處理與對話式 AI)**

語音處理是讓 AI 擁有「耳朵」與「嘴巴」的關鍵技術。隨著大型語言模型 (LLM) 的普及，現在的戰場已經從單純的語音辨識 (ASR) 與語音合成 (TTS)，轉移到強調低延遲、能處理自然打斷的「即時對話代理 (Voice Agents)」。

### 🩸 實戰血淚史：ASR / TTS 落地踩坑指南
如果你正準備踏入語音開發的深坑，請務必先停下來看看這些實務經驗。演算法再好，遇到現場的「背景噪音」、「麥克風收音距離」與「詭異的口音」，模型一樣會崩潰。

> **💡 開發者的真心話：數據質量決定一切**  
> 在語音領域，**「垃圾進，垃圾出」** 的現象比影像或文本更嚴重。很多時候，與其花時間去微調模型參數，不如好好去清洗你的音檔數據、處理好降噪 (Noise Reduction) 與 VAD (語音活動偵測)。
> 
> 👉 **必讀實務經驗分享：**
> * **[ASR / TTS 開發避坑指南](https://blog.twman.org/2024/02/asr-tts.html)**：深度解析在真實場景中導入語音技術時，如何評估數據質量與避開常見架構陷阱。
> * **[那些語音處理踩的坑](https://blog.twman.org/2021/04/ASR.html)**：從前端收音到後端辨識，記錄了我們過去在專案落地時滿滿的血淚與實戰心得。

---

### 🚀 核心框架與即時對話技術 (Frameworks & Real-time AI)
要在本地端部署極速的語音模型，或是打造像 ChatGPT Voice 一樣能自然對話的 AI，你需要以下這些前沿框架：

* **[TEN Framework](https://github.com/TEN-framework)** `[2025-05-14]`
  * **核心優勢**：專為即時對話式 AI (Real-time Conversational AI) 打造的強大框架。
  * **解決痛點**：解決了 AI 語音助理最難的「什麼時候該講話」與「什麼時候該閉嘴」的問題。其內建的 **[ten-vad](https://zread.ai/TEN-framework/ten-vad)** (語音活動偵測) 與 **[ten-turn-detection](https://zread.ai/TEN-framework/ten-turn-detection)** (語音輪替偵測)，是實現自然打斷 (Interruption) 與流暢對話的核心組件。

* **[[LiveKit Agents]](https://github.com/livekit/agents)** `[持續更新]` 🔥 *(9.9k+ Stars)*
  * **核心優勢**：**基於 WebRTC 的頂規即時語音/視訊 AI Agent 框架！** 建構於 LiveKit SFU 伺服器之上，原生繼承 WebRTC 的抗弱網、自動重連與極低延遲特性。其最大亮點是內建基於 Transformer 的**「語義輪次檢測 (Semantic Turn Detection)」**與 ML 驅動的**「自適應打斷處理」**，能精準區分真實打斷與背景噪音（如咳嗽、嘆氣），在 500ms 重疊語音下實測精確率達 86%、召回率 100%。
  * **解決痛點 / 推薦場景**：**徹底解決傳統語音 Agent 依賴「固定靜音時長 (VAD)」導致的頻繁搶話與誤判問題。** 擁有極其豐富的生態，內建 64 個插件覆蓋所有主流 LLM、STT、TTS 與 Avatar。原生支援 MCP (Model Context Protocol) 協議與「多 Agent 狀態交接」。非常適合開發企業級全雙工語音客服、多模態視訊面試官、以及需要極低延遲的即時對話虛擬人。
  * **資源**：[🐙 GitHub](https://github.com/livekit/agents) | [🌐 LiveKit 官方網站](https://livekit.io/) | [📦 Plugins 官方目錄](https://github.com/livekit/agents/tree/main/livekit-plugins)
  <details>
  <summary><b>🛠️ 核心架構解析與開箱即用代碼 (點擊展開)</b></summary>

  #### ⚙️ 核心架構：Agent 如何入會
  Agent 以 WebRTC 參與者的身分直接加入房間 (Room)。工作流程分為四核心：
  1. **AgentServer (主進程)**：向 LiveKit 伺服器註冊，負責 Job 調度。
  2. **JobContext**：當使用者進入 Room，伺服器發起調度啟動 Job 子進程。
  3. **AgentSession**：管理 Agent 與終端使用者的互動容器，處理音視訊串流收發。
  4. **Agent**：帶有指令定義與工具 (Tools) 的 LLM 實體。

  #### 🎛️ 三種運行模式 (極度友善的開發體驗)
  支援免配伺服器的終端對話，也支援熱重載：
  * **Console 模式**：`python myagent.py console` (本地麥克風直接測試，免 Server)
  * **Dev 模式**：`python myagent.py dev` (熱重載 + 連接 LiveKit Server)
  * **Prod 模式**：`python myagent.py start` (生產環境優化部署)

  #### 🧩 強大生態與 MCP 支援
  透過 `pip install "livekit-agents[openai,silero,deepgram,cartesia,turn-detector]~=1.4"` 按需安裝。原生支援將 Python 函數轉為 LLM 工具，並可輕易串接 MCP 服務：
  ```python
  from livekit.agents import function_tool, RunContext

  @function_tool
  async def lookup_weather(context: RunContext, location: str) -> str:
      """查詢指定地點的天氣"""
      return f"{location} 的天氣是晴天，25°C"
  🚀 最小啟動範例 (Python)
  Python
  from livekit.agents import Agent, AgentSession, AgentServer, JobContext, RunContext, function_tool
  from livekit.plugins import openai, silero, deepgram, cartesia

  server = AgentServer()

  @server.rtc_session()
  async def entrypoint(ctx: JobContext):
    # 1. 定義 Agent 與工具
    agent = Agent(instructions="你是一個友善的語音助手", tools=[lookup_weather])
    
    # 2. 組合多模態管線 (簡寫語法亦可: stt="deepgram/nova-3")
    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load()
    )
    
    # 3. 啟動會話並加入 WebRTC 房間
    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(instructions="greet the user and ask about their day")
  </details>


* **[[Sherpa-ONNX]](https://github.com/k2-fsa/sherpa-onnx)** `[持續更新]` 🔥
  * **核心優勢**：**端側語音神經網路高鐵！新一代 Kaldi (k2) 團隊打造的終極跨平台推論框架。** 提供 C++、Python、Go、Swift 等 12 種語言 API，並完美適配 iOS、Android、樹莓派、RISC-V 及國產 NPU（如瑞芯微、昇騰）。支援 ASR、TTS 與 VAD，無縫相容 Whisper、Zipformer 等主流模型。
  * **解決痛點 / 推薦場景**：**用工程極致換取部署下限，徹底解決「斷網/低算力」環境的語音 AI 落地難題。** 實現百毫秒級超低延遲流式推論。極度適合智能硬體/IoT（智慧音箱、掃地機器人、車機）、端側 App 離線開發，以及資料絕對不能上雲的隱私敏感場景（醫療病歷、金融會議）。
  * **資源**：[🐙 GitHub](https://github.com/k2-fsa/sherpa-onnx)

* **[[Seeduplex]](https://seed.bytedance.com/seeduplex)** `[閉源商業標竿]` 💎
  * **核心優勢**：**定義「邊聽邊說」的即時互動新高度，字節跳動首發原生全雙工語音大模型！** 已全量部署於「豆包 App」。徹底拋棄依賴 VAD 機械切音的半雙工架構，融合聲學與語義特徵。其具備強大的「聲學專注力」與「動態判停」機制，將打斷響應延遲大幅縮短 300ms，誤回覆率與搶話比例驟降近一半。
  * **解決痛點 / 推薦場景**：**完美解決傳統語音助理「頻繁搶話」、「無法過濾背景噪音」與「不允許思考留白」的三大死穴。** 無論是在吵雜的咖啡廳閒聊、車內導航播報中下達指令，或是英文面試時因卡頓而產生停頓，Seeduplex 都能像真人一樣耐心等待或即刻響應。這是產業界開發高階語音 Agent、即時虛擬陪伴時，必須親自體驗的效能天花板。
  * **資源**：[🌐 官方專案主頁](https://seed.bytedance.com/seeduplex) | 📱 **體驗方式**：下載最新版豆包 App 使用「打電話」功能

  <details>
  <summary><b>📊 核心突破與真實場景實測數據 (點擊展開)</b></summary>

  #### 🛡️ 精準抗干擾 (聲學專注力)
  * **數據表現**：誤回覆率、誤打斷率較半雙工模型**降低 50%**。
  * **實測場景**：在多人重疊對話中，能精準辨識指向自己的指令；能聯動環境音（如背景正在播放景點介紹），結合對話上下文給出貼合的回應，不再被無關人聲輕易觸發。

  #### ⏱️ 動態判停 (快慢有度，收放自如)
  * **數據表現**：搶話比例下降 40%；判停延遲降低約 250ms。整體通話滿意度絕對值提升 8.34%。
  * **實測場景**：
    * **思考留白**：當用戶說話中途卡頓（如構思外語詞彙），模型能感知語義未完成，耐心等待而不誤判結束。
    * **快問快答**：在飛花令等高節奏遊戲中，話音剛落即刻無縫響應。
    * **瞬間打斷**：當用戶說出「等一下」，模型能瞬間停播收聲，無縫轉入聆聽狀態。
  </details>

---

### 🔮 產業洞察與未來趨勢 (Industry Trends)
* **[小米語音首席科學家 Daniel Povey：語音辨識捲完了，下一個機會在哪裡？](https://www.jiqizhixin.com/articles/2025-01-19-4?)** `[2025-01-19]`
  * **推薦理由**：Kaldi 之父 Daniel Povey 的深度觀點。當標準語音辨識的準確率已經逼近人類極限，未來的語音戰場將轉向「語音到語音 (Speech-to-Speech) 的端對端大模型」以及更深層的語意與情緒理解。

---

### 📚 語音底層基礎知識 (Fundamentals)
無論是想自己訓練模型，還是理解音訊底層邏輯，這些資源能幫你打穩基本功：

#### 影音底層原理與技術綜述
* **[音視頻開發基礎入門](https://zhuanlan.zhihu.com/p/577850804)**：徹底搞懂聲音採集、採樣率 (Sample Rate)、量化位元數與碼率等底層物理聲學知識。
* **[萬字語音合成基礎與論文總結](https://mp.weixin.qq.com/s/S9T9fk9THUF3JQRnNuOM7Q)**：從傳統拼接合成到深度學習端對端 TTS 的完整技術脈絡梳理。

---

## Speech-Recognition
**中文語音識別 (Chinese Speech Recognition)**
> 通過語音信號處理和模式識別讓機器自動識別和理解人類的口述。
> [🌐 更多 ASR 資源](https://www.twman.org/AI/ASR)

-----

### 👑 2026 全球開源 ASR 語音辨識模型大比拚 (非中/歐美大廠篇)

在語音辨識領域，目前主要分為兩大陣營：一派是歐美主導的**「Whisper 生態系與巨頭大模型」**，專注於極限吞吐量與串流延遲；另一派則是亞洲大廠針對**「中文語境、方言與複雜環境」**特化的 SOTA 模型。

#### 1. 歐美 AI 巨頭與 Whisper 生態系 (效能與極速)
*解決痛點：極致壓榨推理速度、精準時間戳對齊，以及串流即時辨識。*

| 模型/工具名稱 | 開發團隊/生態 | 💡 核心優勢與解決痛點 | 🚀 推薦適用場景 & 規格標籤 |
| :--- | :--- | :--- | :--- |
| **Sherpa-onnx** | 🌐 **k2-fsa 團隊** | **全能型本地端語音部署神器**。基於 ONNX Runtime，解決了移動端需自行編寫大量 JNI 程式碼的痛點，單一 AAR 即可支援流式 ASR、VAD 與 TTS。 | 離線語音助理、Android/iOS 原生開發<br>`[本地部署]` `[流式識別]` |
| **WhisperX** | 開源社群 | **精準時間戳**：強力對齊字級時間戳，解決原版糊在一起的問題。 | 會議紀錄、自動上字幕 |
| **Distil-Whisper** | 開源社群 | **輕量極速**：模型縮小 49%，速度提升 6 倍，保留 99% 精準度。 | 本地伺服器、英文場景 |
| **Insanely-Fast-Whisper** | 開源社群 | **天下武功唯快不破**：底層優化，推理速度達到令人髮指的地步。 | 海量音檔批次處理 |
| **CarelessWhisper** | 開源社群 | **低延遲串流**：微調 Whisper 實現接近非串流式的精準度。 | 語音助理、直播字幕 |
| **Parakeet-tdt-0.6b-v3** | **NVIDIA** | **吞吐量王者**：1秒轉錄1小時音訊！輝達最強開源模型之一。 | 企業級資料清洗 |
| **Voxtral (Mini 4B)** | **Mistral AI** | **實時對話**：超越 GPT-4o mini 的語音能力，歐洲巨頭首發。 | 整合 LLM 的語音應用 |
| **OpusLM** | **CMU** | **多模態統一**：學術界重磅！統一語音辨識、合成與文字理解。 | AI 研究、多模態系統 |
| **MedASR** | **Google** | **醫療專精**：解決醫學專業術語難以辨識的痛點。 | 醫療院所、數位健康 |
| **[MAI-Transcribe-1](https://microsoft.ai/news/today-were-announcing-3-new-world-class-mai-models-available-in-foundry/)** `[2026-04]` 🔥 | **Microsoft AI** | **25 種語言性能全數超越 Whisper-large-v3**。解決了長音訊轉寫「越播越崩」的語意偏差，批量轉寫速度提升 2.5 倍，且價格僅每小時 0.36 美元，徹底瓦解高昂轉錄成本。 | 全球化會議逐字稿、多人 Podcast 轉錄、多語言客服系統<br>`[性價比之王]` `[超越Whisper]` |
| **[MAI-Voice-1](https://microsoft.ai/news/today-were-announcing-3-new-world-class-mai-models-available-in-foundry/)** `[2026-04]` | **Microsoft AI** | **1 秒生成 60 秒極致自然語音**。針對長時間敘事優化，完美保留音色一致性與豐富情感，並支援「秒級」小樣本語音克隆。 | 互動式虛擬助理、長篇有聲書製作、遊戲 NPC 語音<br>`[超低延遲]` `[高保真克隆]` |

* **[[Microsoft Foundry-Local Edge ASR (Nemotron-0.6B)]]** `[2026-04]` 🔥
  * **核心優勢**：**打破流式識別準確率崩盤魔咒！670MB 體積極致壓縮，純 CPU 實現 6 倍即時超高速推理。** 微軟最新釋出的端側語音黑科技，基於 NVIDIA Nemotron-0.6B 的「快取感知 (cache-aware)」架構進行深度改造。它採用創新的 **int4-k-quant 混合精度量化**（依據權重重要性保留注意力機制的高精度，並大幅壓縮中間層），將模型體積暴砍 73% 的同時，詞錯率 (WER) 僅微幅退化 0.17 個百分點 (8.20%)。徹底解決了多數開源模型（如 Qwen3-ASR）在切換為流式切片處理時，因前後文斷裂導致錯誤率翻倍的致命缺陷。
  * **解決痛點 / 推薦場景**：**完美解決了傳統高精度 ASR（如 Whisper Large）「極度吃顯存、會卡頓」與雲端 API「延遲高、隱私外洩風險」的兩難痛點。** 演算法延遲低至無感的 0.56 秒，讓低階筆記型電腦或記憶體受限的邊緣設備 (Edge Devices) 也能流暢運行商用級別的即時語音識別。是打造**完全離線的隱私語音助手**、**本地高併發會議即時字幕**、以及**無網環境穿戴式裝置 (Wearables)** 的工業級端側首選。

---

#### 2. 亞洲頂尖開源 ASR 模型 (中文語境特化篇)
*如果你處理的音訊包含大量複雜的中文方言、中英夾雜，或是極具挑戰性的長時段錄音，以下模型目前處於領先地位。（註：注重地緣資安合規的專案，請自行評估導入風險）*

| 模型名稱 | 開發團隊 | 💡 核心優勢與突破點 | 🚀 推薦場景 |
| :--- | :--- | :--- | :--- |
| **FireRedASR2S** | 🇨🇳 **小紅書** | **SOTA 工業全能系統**：在複雜口音與噪音場景下辨識率極強悍。 | 短影音、內容監控 |
| **Qwen3-ASR** | 🇨🇳 **阿里巴巴** | **多語種霸主**：吊打原生 Whisper，支援高達 52 種語言和方言。 | 出海企業、多語客服 |
| **VibeVoice-ASR** | 🇨🇳 **微軟亞洲** | **拒絕切片**：64K 超長窗口，一次吞下 60 分鐘音訊並吐出結構化結果。 | 長篇演講、一小時會議 |
| **Fun-ASR** | 🇨🇳 **阿里達摩院** | **小參數大能量**：0.8B 效能直逼 12B，支援離線轉寫 SDK。 | 邊緣運算、本地部署 |

---

### 🔥 2025-2026 最新 ASR 模型資源庫 (完整收錄)

#### 🇨🇳 亞洲與中文特化模型 (Chinese & Asian Languages)

* **[[StepAudio 2.5 ASR]](https://platform.stepfun.com/docs/zh/guides/models/stepaudio-2.5-asr)** `[2026-04-24]` 🔥
  * **核心優勢**：**導入 LLM MTP 並行預測技術的 ASR 效率革命，推理速度暴增 400%！** 階躍星辰 (StepFun) 發布的新一代語音大模型，採用 Audio Encoder + LLM + MTP-5 深度融合架構。它從底層打破了傳統 ASR 自回歸逐字生成的效率瓶頸，實時比 (RTF) 低至驚人的 0.0053（轉寫 1 小時音訊僅需約 19 秒），且 API 呼叫成本狂降 80% (僅約 0.15 RMB/小時)，展現出破壞性的極致性價比。
  * **解決痛點 / 推薦場景**：**徹底解決長音檔「切片導致語意斷裂」與「商業大批量轉寫成本過高」的雙重痛點。** 復用大語言模型原生的 32K 上下文視窗，完美支援高達 30 分鐘連續音檔「一刀未剪」的端到端完整轉寫，解決了長文本識別常見的精度逐級衰減問題。極度適合用於**海量會議紀錄歸檔**、**長篇 Podcast/採訪逐字稿**、**企業客服錄音大數據質檢**，以及需要精準還原口語特徵與中英夾雜的複雜語境。（⚠️ **實戰避坑指南**：實測發現在「上傳音檔模式」處理非標準或特定音源時，偶有辨識不到清晰語音的穩定性波動，建議企業在正式上線前針對私有場景進行灰度測試）。
  * **資源**：[🌐 線上體驗中心](https://www.stepfun.com/studio/audio?tab=speech-recognition) | [📖 API 整合文件](https://platform.stepfun.com/docs/zh/step-plan/integrations/audio-api) | [📊 Model Card](https://stepaudiollm.github.io/step-audio-2.5-asr/model-card/)

* **[[FireRedASR2S]](https://github.com/FireRedTeam/FireRedASR2S)** `[2026-02-12]` 🔥
  * **核心優勢**：**小紅書開源的工業級「四合一」全能語音系統，SOTA 級的方言與歌聲辨識霸主。** 首創將 VAD (語音活動檢測)、LID (語種路由)、ASR (語音辨識) 與 Punc (標點預測) 完美整合的端到端管線，徹底消除傳統模組拼湊導致的前後級「誤差級聯」問題。
  * **解決痛點 / 推薦場景**：**直擊工業界「長音頻漏切」、「方言/中英夾雜」與「背景噪音干擾」三大痛點。** 憑藉近 20 萬小時高質量數據，精準拿捏 100+ 語言與 20+ 中國方言（粵/吳/閩/藏語等），甚至連高難度的「唱歌歌詞」都能極致還原。極度適合高併發的短影音自動字幕、Podcast 長音頻逐字稿、以及複雜環境下的客服質檢系統。
  * **模組亮點**：
    * **FireRedVAD**：僅 0.6M 參數 (~2.2MB)，改用人工標註聲學庫，抗噪極強，F1 分數 (97.57%) 完勝 Silero 與 WebRTC。
    * **FireRedASR2**：提供 LLM 版 (8B+) 與 AED 版 (1B+)。AED 版新增後置 CTC 分支，兼顧高精度與「詞級時間戳」。
    * **FireRedLID & Punc**：層級標籤精準路由語種，標點預測模型經 185 億漢字微調，大幅提升文本閱讀性。
  * **資源**：[🐙 GitHub](https://github.com/FireRedTeam/FireRedASR2S) | [🤗 Hugging Face](https://huggingface.co/FireRedTeam) | [🌐 線上 Demo](https://huggingface.co/spaces/FireRedTeam/FireRedASR)

  <details>
    <summary><b>📊 核心實測對標數據 & 💻 本地部署指南 (點擊展開)</b></summary>
    
    #### 🏆 ASR 綜合表現 (字錯誤率 CER%，越低越好)
    | 測試集類別 | FireRedASR2-LLM | FireRedASR2-AED | Doubao-ASR | Qwen3-ASR | Fun-ASR |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | **4個普通話集 (平均)** | **2.89** | 3.05 | 3.69 | 3.76 | 4.16 |
    | **19個方言集 (平均)** | **11.55** | 11.67 | 15.39 | 11.85 | 12.76 |
    | **唱歌歌詞 (opencpop)** | **1.12** | 1.17 | 4.36 | 2.57 | 3.05 |

    #### 🛡️ VAD 效能對比 (FLEURS-VAD-102 多語言集)
    | 指標 | FireRedVAD | Silero-VAD | TEN-VAD | FunASR-VAD | WebRTC-VAD |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | **F1 Score (%) ↑** | **97.57** | 95.95 | 95.19 | 90.91 | 52.30 |
    | **誤報率 FAR (%) ↓** | **2.69** | 9.41 | 15.47 | 44.03 | 2.83 |
    | **漏報率 MR (%) ↓** | 3.62 | 3.95 | 2.95 | **0.42** | 64.15 |
    *(註：FireRedLID 準確率 97.18% 亦優於 Whisper 的 79.41%；標點預測 FireRedPunc F1 達 78.90%，遠超 FunASR-Punc)*

    #### 🚀 企業級部署與加速指南
    * **硬體與音頻規範**：必須為 `16kHz`, `16-bit`, 單聲道 PCM WAV。AED 版建議音頻 `≤60秒`，LLM 版建議 `≤40秒`。
    * **極速推理支援**：原生支援 **vLLM**，且 AED 版獲 NVIDIA 貢獻 **TensorRT-LLM** 加速，單卡 H20 推理比 PyTorch 快 **12.7倍**。

    **💻 Python 開箱即用 API：**
    ```python
    from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig

    # 默認配置自動串聯 VAD → LID → ASR → Punc 四大模組
    config = FireRedAsr2SystemConfig()  
    asr_system = FireRedAsr2System(config)

    # 輸出結構化 JSON：含標點文本、毫秒級時間戳、語種、置信度、詞級切分
    result = asr_system.process("assets/hello_zh.wav")
    print(result)  

</details>

* **[[VibeVoice Family]](https://github.com/microsoft/VibeVoice)** `[2026-01-30]` 🔥 *(35k Stars)*
  * **核心優勢**：**開源語音界的「桌子翻轉者」，首創 7.5Hz 超低幀率語音 Tokenizer 技術！** 微軟出品的顛覆性系列模型，包含 ASR (7B)、TTS (1.5B) 與 Realtime (0.5B)。其核心突破在於將語音壓縮率提升 99%，在極低計算量下保留高品質聲學細節。Realtime 版本首音延遲僅 **300ms**，效能直接對標 GPT-4o 語音模式。
  * **解決痛點 / 推薦場景**：**徹底瓦解商業語音 API 每百萬 Token 收費 64 美元的定價邏輯。** 完美解決了企業在導入語音 AI 時擔心的「資料外流」、「API 限速」與「昂貴成本」三大痛點。非常適合建構本地化的企業語音輸入法、7×24 小時無人語音客服，以及對延遲極度敏感的即時語音翻譯系統。
  * **資源**：[🐙 GitHub](https://github.com/microsoft/VibeVoice) | [📄 ASR 技術報告 (arXiv)](https://arxiv.org/pdf/2601.18184) | [🤗 HF Transformers 整合](https://huggingface.co/microsoft)

- **[[Qwen3-ASR + vLLM 高併發部署]](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B)** `[2026-04]` 🔥 *(取代原本 2026-01-30 的舊版目)*
  - **核心優勢**：**高併發直接拉滿的工業級語音辨識神作**。將開源頂規的 Qwen3-ASR (支援 52 種語言及閩南語、粵語等複雜中文方言) 結合 vLLM 推理引擎，實現極低延遲（20 字音訊僅需 ~300ms）與超高吞吐量，徹底解放 GPU 算力。
  - **解決痛點 / 推薦場景**：完美解決傳統開源模型面對大量並發請求時容易卡死或延遲過高的痛點。官方提供標準化 Docker 映像檔 (`qwenllm/qwen3-asr`)，大幅降低環境配置門檻，是打造**高併發自動上字幕**、**企業級多語種客服**的絕對首選。
  - **資源**：[🤖 ModelScope 下載](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B) | [📦 Docker 映像檔](https://hub.docker.com/r/qwenllm/qwen3-asr) | [📝 vLLM 部署指南參考](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B/summary)
  <br>`[高併發首選]` `[極低延遲]` `[中文方言霸主]`

- **[2026-01-30] Qwen3-ASR**
  - **說明**：支援52 種語言和方言，吊打 Whisper。
  - **資源**：[🤗 HuggingFace DEMO](https://huggingface.co/spaces/Qwen/Qwen3-ASR) | [📝 公眾號解讀](https://mp.weixin.qq.com/s/-7gm2BstDVxTkJ6lD3Znmg)

- **[2025-12-16] Fun-ASR**
  - **說明**：主打極致性價比與輕量化，僅需極低顯存即可在本地端流暢運行 0.8B 模型。內建完善的離線轉寫 SDK，適合邊緣運算設備或封閉內網環境部署。
  - **資源**：[🐙 GitHub](https://github.com/FunAudioLLM/Fun-ASR) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1984310683358217029) | [📝 微信微調教學](https://mp.weixin.qq.com/s/M1vGqFZV5MWREkSyx2-ITw)

- **[2025-12-15] GLM-ASR**
  - **說明**：解決複雜聲學環境、方言辨識以及低音量語音。
  - **資源**：[🐙 GitHub](https://github.com/zai-org/GLM-ASR) | [🤗 HuggingFace](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1983951645055419349)

- **[2025-04-02] Dolphin**
  - **說明**：Large-Scale ASR Model for Eastern Languages。
  - **資源**：[🐙 GitHub](https://github.com/DataoceanAI/Dolphin) | [📄 arXiv](https://arxiv.org/abs/2503.20212)

* **[[SenseVoice]](https://github.com/FunAudioLLM/SenseVoice)** `[持續更新]` 🔥
  * **核心優勢**：**Whisper 的最強本地端平替，ASR + 情感識別 + 音訊事件檢測「三合一」大模型！** 阿里通義實驗室推出的開源神作。推理速度比 Whisper-Large 快高達 15 倍（10 秒音訊僅需 70ms），支援 50+ 語言。不僅能精準轉錄文字，還能同步辨識說話者的情緒（開心、緊張等）與環境背景音（掌聲、音樂等）。
  * **解決痛點 / 推薦場景**：**徹底解決本地端語音模型「太慢」、「太吃硬體」以及「缺乏語境感知」的三大痛點。** 硬體門檻極低，Small 模型僅需 8GB 記憶體，純 CPU 也能流暢運行。極度適合開發完全離線的隱私語音輸入法（如 Mac 上的 Murmur）、低成本長照輔助設備，或是需要感知用戶情緒的虛擬 AI 伴侶。
  * **資源**：[🐙 GitHub](https://github.com/FunAudioLLM/SenseVoice) | [🌐 專案主頁](https://funaudiollm.github.io/) | [📝 深度解讀與實測](https://mp.weixin.qq.com/s/_bfFhuG8h_sPQ8L_MEKTJw)
  - **[2024-07-03] SenseVoice**
    - **說明**：阿里開源，支援偵測掌聲、笑聲等非語音事件。
    - **資源**：[🌐 Project](https://funaudiollm.github.io/) | [📝 中文解讀](https://mp.weixin.qq.com/s/q-DyyAQikz8nSNm6qMwZKQ)

  <details>
  <summary><b>💻 30 秒快速上手程式碼 & 真實落地案例 (點擊展開)</b></summary>

  #### 🏆 輕量化真實落地案例
  * **Murmur**：完全離線的 macOS 免費語音輸入法，採用 Native Swift 呼叫 SenseVoice ASR。
  * **Voily**：結合 MLX 框架在 Apple Silicon 本地端運行的 AI 語音輸入法，實現極低延遲的流式輸出。
  * **極低成本終端方案**：開發者實測使用「8GB 二手手機 + SenseVoice + Qwen 0.6B」，成本甚至低於樹莓派，即可打造強大的邊緣語音 AI 設備。

  #### 🚀 Python 開箱即用 API (純 CPU 可跑)
  無需複雜配置，一行指令安裝 `pip install funaudio`，即可同時輸出文字與情感標籤：

  ```python
  from funaudio import SenseVoiceSmall

  model = SenseVoiceSmall()
  audio_path = "your_audio.wav"
  result = model.transcribe(audio_path)

  print(f"📝 轉錄文本: {result['text']}")
  print(f"🎭 情感標籤: {result.get('emotion', 'N/A')}")

(註：命令列模式亦支援直接輸出 JSON 格式：funaudio --model sensevoice-small --file test.wav --output-format json)

  </details>

---

#### 🌐 國際巨頭與創新架構 (Global Tech & Innovations)

* **[[Voxtral Realtime (4.4B)]](https://arxiv.org/pdf/2602.11298)** `[2026-02-28]` 🔥
    * **核心優勢**：**打破「低延遲與高精度」互斥魔咒的開源即時 ASR 霸主**。由 Mistral AI 採用 Apache 2.0 開源，創新導入全因果音訊編碼器 (Causal Audio Encoder) 與 Ada RMS-Norm 架構。在 **480ms 的亞秒級延遲下，辨識精準度直接打平 Whisper 離線模型**；更在工程端深度整合 `vLLM`，支援異構 KV 快取與 WebSocket 全雙工即時推理。
    * **解決痛點 / 推薦場景**：**徹底解決傳統離線模型（如 Whisper 滑動窗口）硬改為流式推理時精準度暴跌的痛點，以及生產環境中長文本泛化能力弱的瓶頸**。單一模型即可動態切換延遲檔位（80ms\~2400ms），支援 13 種語言。是打造**高併發即時字幕**、**同聲傳譯 (AI 口譯)**、**低延遲全雙工語音助理**的工業級首選。<br>`[亞秒級延遲]` `[vLLM原生支援]` `[Apache 2.0 完全開源]`
    * **資源**：[📄 官方論文](https://arxiv.org/pdf/2602.11298) | [🐙 GitHub (Mistral AI)](https://github.com/mistralai)

* **[[Cohere Transcribe]](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)** `[2026-03-28]` 🔥
  * **核心優勢**：**20 億參數塞進瀏覽器，準確率正式碾壓 Whisper Large v3！** Cohere 釋出的 2B 參數 SOTA 語音模型 (cohere-transcribe-03-2026)，採用 Conformer 編碼器與輕量 Transformer 解碼器架構。它以 5.42% 的平均詞錯率 (WER) 登頂 HuggingFace Open ASR 排行榜，超越 OpenAI Whisper Large v3 (6.41%)。最震撼的是，透過 WebGPU 與 ONNX Runtime，它能直接在網頁瀏覽器內免安裝本地執行，1 小時的錄音僅需約 100 秒即可轉錄完畢 (約 36 倍即時倍率)。
  * **解決痛點 / 推薦場景**：**徹底解決機密音檔「上傳雲端」的隱私疑慮與伺服器部署的高昂算力成本。** 由於完全在終端瀏覽器本地運行，音頻不上傳伺服器，是處理敏感採訪、內部會議等極機密資料的最佳開源方案 (Apache 2.0 可商用授權)。模型也在 vLLM 獲得首日支援與最佳化，語音推理吞吐量最高提升 2 倍，企業自託管部署同樣強悍。**⚠️ 避坑指南**：目前版本不支援時間戳與說話人分離，需手動指定語言，且在噪音環境下易產生幻聽，強烈建議搭配 VAD 前端處理使用。
  * **資源**：[🐙 官方模型](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | [🌐 官方介紹](https://cohere.com/blog/transcribe) | [📦 ONNX 瀏覽器版本](https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX) | [⚙️ vLLM 優化 PR](https://github.com/vllm-project/vllm/pull/38120)

- **[2026-02-04] Voxtral (Mistral)**
  - **說明**：Mistral 開源語音模型 Voxtral Mini 4B Realtime；在 480ms 延遲下英語短音頻 WER 為 8.47%，與離線 Whisper（8.39%）幾乎持平。GPT-4o mini Transcribe 均被歸類為"實時API"模型，同類流式模型 Nemotron Streaming 在 560ms 延遲下 WER 為 9.59%，差距明顯。支援的 13 種語言：英語、中文、西班牙語、法語、德語等。
  - **資源**：[🤗 HuggingFace](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | [📝 公眾號解讀](https://mp.weixin.qq.com/s/5fG_xOrPIsXCs5sNiFidMQ) | *(2025-07-16 舊版：[Small 24B](https://huggingface.co/mistralai/Voxtral-Small-24B-2507) / [Mini 3B](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) / [📝 中文解讀](https://zhuanlan.zhihu.com/p/1928945056955471125))*

- **[2025-12-23] MedASR**
  - **說明**：Google 發布醫學語音辨識模型。
  - **資源**：[🤗 HuggingFace](https://huggingface.co/google/medasr)
- **[2025-11-15] Omnilingual-ASR**
  - **資源**：[🐙 GitHub](https://github.com/facebookresearch/omnilingual-asr) | [🌐 DEMO](https://aidemos.atmeta.com/omnilingualasr/language-globe)

- **[2025-08-10] Canary-1b-v2**
  - **說明**：NVIDIA 發布多語種語音 AI 開放資料集與模型。
  - **資源**：[🤗 HuggingFace](https://huggingface.co/nvidia/canary-1b-v2) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1952436345222993067)

- **[2025-08-08] Parakeet-tdt-0.6b-v3**
  - **說明**：1秒轉錄1小時音訊！輝達最強開源模型。
  - **資源**：[🤗 HuggingFace](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | [📝 媒體報導](https://hk.finance.yahoo.com/news/1%E7%A7%92%E8%BD%89%E9%8C%841%E5%B0%8F%E6%99%82%E9%9F%B3%E8%A8%8A-%E8%BC%9D%E9%81%94%E9%87%8D%E7%A3%85%E9%96%8B%E6%BA%90%E8%AA%9E%E9%9F%B3%E8%AD%98%E5%88%A5%E6%9C%80%E5%BC%B7%E6%A8%A1%E5%9E%8Bparakeet-075846970.html)

- **[2025-07-02] OpusLM**
  - **說明**：CMU 發布統一語音辨識、合成、文字理解的大模型。
  - **資源**：[🤗 HuggingFace](https://huggingface.co/espnet/OpusLM_7B_Anneal) | [📝 中文解讀](https://mp.weixin.qq.com/s/XCgBTgfOs8y_fFFEEMrW-w)

- **[2025-05-06] VITA-Audio**
  - **說明**：快速交錯跨模態令牌生成。
  - **資源**：[📚 DeepWiki](https://deepwiki.com/VITA-MLLM/VITA-Audio) | [📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/2505.03739)

---

#### ⏱️ Whisper 變體與串流應用工具 (Streaming & Tools)

* **[[Sherpa-ONNX Android Agent]](https://github.com/coder-brzhang/funasr-agent)** `[2026-04-18]` 🔥
  * **核心優勢**：**地表最強 Android 離線語音助手實作範例，整合流式 ASR 與 Silero VAD**。利用 `sherpa-onnx` 框架與 `Paraformer` 雙語模型，實現完全「純離線」的邊說邊出字體驗。僅需 15MB 的 AAR 核心庫與 int8 量化模型，即可在安卓中端設備達成 \<300ms 的極致響應，並具備模糊指令匹配能力。
      * **解決痛點 / 推薦場景**：**解決了離線環境下語音助手「聽不懂」與「反應慢」的雙重挑戰**。透過 VAD 自動人聲檢測免除按鈕操作，是打造隱私優先、低功耗、免聯網 AI 語音助理（如智慧家居中樞、車載語音、Wear OS 設備）的教科書級參考。
  * **資源**：[🐙 GitHub](https://github.com/coder-brzhang/funasr-agent) | [📝 深度實作教學](https://mp.weixin.qq.com/s/DOm_hg6DWA_OjcsLuUQ9Hw) | [📄 Paraformer 論文](https://arxiv.org/abs/2206.08317)

- **[2025-08-29] WhisperLiveKit**
  - **說明**：讓即時語音轉寫絲滑得不像話的神器。
  - **資源**：[🐙 GitHub](http://github.com/QuentinFuxa/WhisperLiveKit) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1944712252512010607)

- **[2025-08-27] CarelessWhisper**
  - **說明**：微調 Whisper 實現低延遲串流識別，效果接近非串流式。
  - **資源**：[🐙 GitHub](https://github.com/tomer9080/CarelessWhisper-streaming) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1977136140139141051)

- **[2025-06-06] speakr**
  - **說明**：開源轉錄工具，支援 AI 生成內容。
  - **資源**：[🐙 GitHub](https://github.com/murtaza-nasir/speakr) | [📝 中文解讀](https://cloud.tencent.com/developer/news/2645205)

---

### 📦 經典模型與開發套件庫 (Classic Toolkits)

- **Whisper Family (OpenAI)**
  - [**Whisper**](https://openai.com/research/whisper): OpenAI 開源準確率最高的通用模型。
  - [**WhisperLive**](https://github.com/collabora/WhisperLive): 免費即時語音轉文字工具。
  - [**Distil-Whisper**](https://github.com/huggingface/distil-whisper): 輕量級 AI 的強大力量。
  - [**Insanely-Fast-Whisper**](https://github.com/Vaibhavs10/insanely-fast-whisper): 超快速辨識腳本。
    - [📝 公眾號解讀 | 2.5 小時音訊只需 98 秒轉錄](https://mp.weixin.qq.com/s/D2qBINl2m45IQsG5jJ08WA)
  - [**WhisperX**](https://github.com/m-bain/whisperX): 強化的時間戳記與說話者識別。
  - [**Fine-tune Whisper**](https://huggingface.co/blog/zh/fine-tune-whisper): 微調教學。

- **FunASR (阿里達摩院)**
  - [Github](https://github.com/alibaba-damo-academy/FunASR) | [離線轉寫 SDK](https://zhuanlan.zhihu.com/p/642807244)

- **WeNet (58同城)**
  - [Paper](https://arxiv.org/pdf/2102.01547.pdf) | [落地方案](https://zhuanlan.zhihu.com/p/573133117)

- **Other Toolkits**
  - [**PaddleSpeech**](https://github.com/PaddlePaddle/PaddleSpeech)
  - [**Speech Brain**](https://github.com/speechbrain/speechbrain)
  - [**Kaldi 2 (k2)**](https://github.com/k2-fsa/k2)

<details>
<summary><b>🎙️ 2020/03-2021/01 開發心得：ASR 語音辨識的拓荒與踩坑</b></summary>
<br>

語音辨識技術（Automatic Speech Recognition, ASR / Speech To Text, STT），其目標是讓電腦自動將人類語音轉換為相應的文字。這是一門極度跨領域的深水區，涵蓋了訊號處理、圖型識別、機率論、發聲與聽覺機理以及人工智慧。*(註：ASR 的重點是辨識「內容說了什麼」，這與辨識「是誰說的」說話人辨識 / Speaker Verification 完全不同！)*

<blockquote>
  <b>🛠️ 框架與 API 的神農嚐百草</b><br>
  當時為了搞定 ASR，跟小夥伴們幾乎把市面上的方案全測過了一輪！我們嘗試過 NEMO、Kaldi、MASR、VOSK、wav2vec，也串接過 Google、Azure 等商用 API，更別說後來陸續冒出來的 SpeechBrain、出門問問的 WeNet 跟騰訊 PIKA 等。每一種演算法架構都各有優缺點，實際落地的效果也是如人飲水，冷暖自知。
</blockquote>

<blockquote>
  <b>📊 數據痛點：找不到靠譜的「台灣口音」</b><br>
  搞語音辨識，聲學模型 (AM) 搭配語言模型 (LM) 是基本功，但最大的死穴在於「開源數據庫」。目前公開已知可訓練的中文數據（如：<code>Magic-Data_Mandarin-Chinese-Read-Speech-Corpus</code>、<code>aidatatang</code>、<code>aishell-1</code> 到 <code>aishell-4</code>）大約有 2000 多小時。但可惜的是，這些幾乎全是<b>中國發音與用語</b>，至今仍缺乏較靠譜的台灣在地數據。而且說實話，若真想達到「商用級別」，訓練數據量至少得破萬小時才算及格。
</blockquote>

<b>💡 應用場景與進階延伸</b>
<ul>
  <li><b>基礎落地</b>：語音撥號、語音導航、室內智慧家電控制、語音文件檢索、簡單的聽寫資料錄入等。</li>
  <li><b>高階複合應用</b>：將 ASR 結合其他 NLP 技術（如機器翻譯）與語音合成（TTS）技術，即可構建出如「即時語音到語音的翻譯 (Speech-to-Speech Translation)」等更加複雜且強大的應用。</li>
</ul>

</details>

---

### 🛠️ ASR 後處理與文本糾錯 (ASR Post-Processing & Correction)

* **[[Generative-Annotation-NEC (SS+GL)]](https://github.com/L6-NLP/Generative-Annotation-NEC)** `[2026-04-18]` 🔥
  * **核心優勢**：**華為開源的 ASR 命名實體糾錯 (NEC) 終極方案，徹底解決「音似形異」的轉寫災難**。捨棄傳統依賴「文字拼寫相似度」的編輯距離法，創新提出 SS+GL 架構：先透過「聲音指紋 (Speech-based Selection)」檢索正確實體，再利用大模型進行「生成式標註 (Generative Labeling)」。能精準將「米德仲尼」糾正為「Midjourney」、「01X」糾正為「靈耀X」。
  * **解決痛點 / 推薦場景**：**完美解決通用 ASR（如 Whisper）或第三方雲端語音 API，在遇到專業術語、最新科技詞彙或中英夾雜時辨識崩壞的痛點**。採用「生成後糾正」的解耦架構，**隨插即用**，無須改動底層 ASR 引擎。是打造**高精度醫療/法律會議紀錄**、**企業私有領域語音知識庫**，以及**防過糾錯 (Anti-Overcorrection) 文本清洗管線**的工業級利器。
  * **資源**：[🐙 GitHub (程式碼與資料集)](https://github.com/L6-NLP/Generative-Annotation-NEC) | [📄 官方論文](https://arxiv.org/pdf/2508.20700)

---

## Speaker-Recognition
**🗣️ Speaker Recognition (中文語者與聲紋識別)**
> 通過聲音判別說話人身份的技術 (聲紋特徵)。
> [🌐 更多資源](https://www.twman.org/AI/ASR/SpeakerRecognition)

> **💡 核心觀念**：語音辨識 (ASR) 是破解「說了什麼」，而聲紋識別 (Speaker Recognition) 則是破解「**是誰說的**」。透過提取聲音中的生物特徵（聲紋），實現說話人身份的驗證與辨識。
> 👉 [🌐 更多資源：TWMAN 聲紋識別技術總結](https://www.twman.org/AI/ASR/SpeakerRecognition)

### 1. 主流開源框架與模型 (Open Source Frameworks)
* **Wespeaker**：目前產業界極受歡迎的生產級聲紋辨識開源工具包。[📄 AlphaXiv 論文](https://www.alphaxiv.org/zh/overview/2210.17016v2) | [📝 v1.2.0 發布說明](https://zhuanlan.zhihu.com/p/645726183)
* **SincNet**：一種直接處理原始音訊波形 (Raw Waveform) 的深度學習架構，能有效提取具備物理意義的聲學特徵。[📄 AlphaXiv 論文](https://www.alphaxiv.org/zh/overview/1808.00158v3)

### 2. 實戰教學與開源資料集 (Tutorials & Datasets)
* **實作與原理指南**：
  * [ASV-Subtools 聲紋識別實戰](https://speech.xmu.edu.cn/2022/1124/c18207a465302/page.htm)
  * [深度學習在聲紋識別中的應用](https://yutouwd.github.io/posts/600d0d5d/)
  * [聲紋識別原理科普](https://www.zhihu.com/question/30141460)
  * [相關聲紋識別介紹匯整](http://xinguiz.com/category/#/声纹识别)
* **學術資源與資料集**：
  * [CN-Celeb-AV 多模態資料集](https://zhuanlan.zhihu.com/p/647786644)：極具挑戰性的真實場景中文聲紋庫。
  * [ICASSP 2023 說話人識別方向論文合集（一）](https://zhuanlan.zhihu.com/p/645560614)
  * [提高聲紋辨識正確率 更添防疫新利器 (國網中心 NCHC)](https://www.nchc.org.tw/Message/MessageView/3731?mid=43)

---

<details>
<summary><b>🗓️ 2020/03-2020/08 開發心得：聲紋識別的從零到一與踩坑實錄</b></summary>
<br>
投入約 150 天。通常我們是怎樣開始一個 AI 專案的研究與開發？

<blockquote>
  <b>🔍 R&D 前期調研 SOP (約耗時 30 天)</b><br>
  首先會盡可能把 3 年內的學術論文或比賽的 SOTA 都查過一輪，分工閱讀找到相關的數據集和開源實作。同時，我們會去盤點目前已有相關產品的公司（含新創）以及他們提交的專利（透過 Google Patents, Papers with Code, arXiv 等）。在聲紋識別這塊，對岸有非常多的新創公司，例如<b>「國音智能」</b>，在我們的研發過程中就一直被當作標竿目標。
</blockquote>

<blockquote>
  <b>🚧 數據獲取的「地理限制」與預處理地獄</b><br>
  在分享實驗結果前，必須先警告後人避免踩坑：上述很多中文聲紋數據集都放在對岸的百度雲盤等空間，<b>而百度是直接封鎖台灣 IP 的</b>，所以你打不開是非常正常的！另外，像 <code>VoxCeleb</code> 這種神級數據庫是被切成 7 份的，下載完再合併就要花上不少時間（相比之下 <code>aishell</code>、<code>CMDS</code>、<code>TIMIT</code> 就相對好處理多了）。
</blockquote>

<blockquote>
  <b>🧠 技術架構總結與 Kaldi 泥沼</b><br>
  聲紋技術的發展脈絡可簡單總結為三大核心：<br>
  1. <b>向量抽取 (Vector Extraction)</b>：i-vector, d-vector, x-vector 等。<br>
  2. <b>模型架構與調參</b>：CNN, ResNet 等深度學習架構。<br>
  3. <b>評分方式 (Scoring)</b>：LDA, PLDA (Probabilistic Linear Discriminant Analysis) 等組合。<br>
  我們當時也使用了 <code>Kaldi</code> 內附的功能，光是跟 Kaldi 搏鬥就投入了極大的時間和精力！其實跟 NLP 相比，聲紋識別雖然數據集難搞，但好處是聲音可以自行用程式「加工」做切割合併（Data Augmentation）。因為真實場景錄音通常很短，還得處理「非註冊聲紋 (Open-set)」的拒絕判定，前前後後在數據搭配評分模式上花費了龐大心血，是個不折不扣的大工程。
</blockquote>

<hr>

<b>📐 必備技術指標字典 (Evaluation Metrics)</b>
在聲紋領域，你必須看懂以下指標才能評估模型好壞：
<ul>
  <li><b>FRR (False Rejection Rate, 錯誤拒絕率)</b>：同類的兩人被系統誤判為「不同類」。FRR 為誤判案例在所有同類匹配案例中的比例。（把主人擋在門外）</li>
  <li><b>FAR (False Acceptance Rate, 錯誤接受率)</b>：不同類的兩人被系統誤判為「同類」。FAR 為接受案例在所有異類匹配案例中的比例。（放小偷進門）</li>
  <li><b>EER (Equal Error Rate, 等錯誤率)</b>：調整閥值 (Threshold)，當 FRR = FAR 時的數值稱為等錯誤率。<b>EER 越低，模型越強。</b></li>
  <li><b>ACC (Accuracy, 準確率)</b>：ACC = 1 - min(FAR + FRR)。</li>
  <li><b>ROC 曲線</b>：描述 FAR 和 FRR 間變化的曲線，X 軸為 FAR，Y 軸為 FRR。</li>
  <li><b>閥值 (Threshold)</b>：當系統計算出的相似度分數超過此閥值，才做出「接受 / 確認本人」的決定。</li>
</ul>

<b>⚡ 效能與速度指標</b>
<ul>
  <li><b>實時比 (RTF, Real Time Factor)</b>：衡量提取時間跟音訊時長的關係。例如：1 秒的運算能處理 80 秒的音訊，實時比就是 1:80。</li>
  <li><b>驗證比對速度</b>：平均每秒伺服器能進行的聲紋比對次數。</li>
</ul>

</details>

---

## Speech-Enhancement
**🎧 Speech Enhancement (中文語音增強與去噪)**

> **💡 核心觀念**：從含雜訊的複雜環境音中，精準提取出純淨的人聲（語音信號）。這在語音辨識（ASR）的前處理中，是決定辨識率成敗的關鍵第一步。
> 👉 [🌐 更多資源：TWMAN 語音增強總結](https://www.twman.org/AI/ASR/SpeechEnhancement) | [🤗 線上 DEMO 體驗 (Meta Denoiser)](https://huggingface.co/spaces/DeepLearning101/Speech-Quality-Inspection_Meta-Denoiser)

### 1. 前沿開源去噪模型與框架
* **[ClearVoice](https://github.com/modelscope/ClearerVoice-Studio)** `[2024-12-07]` 🔥
  * **核心優勢**：整合了語音增強與多人語者分離技術，能有效對付極度棘手的**雞尾酒會效應（Cocktail Party Effect）**。
  * **推薦場景**：多人視訊會議紀錄的聲軌拆分、吵雜戶外採訪音檔的極限人聲修復與提取。[🤗 官方 Demo](https://huggingface.co/spaces/alibabasglab/ClearVoice) | [📝 中文原理解讀](https://zhuanlan.zhihu.com/p/18109659892)
* **[Meta Denoiser](https://github.com/facebookresearch/denoiser)**
  * **核心優勢**：Facebook AI Research (FAIR) 推出的即時語音增強開山之作，奠定了深度學習在即時降噪領域的基礎。[📄 AlphaXiv 論文](https://www.alphaxiv.org/abs/2006.12847v3)

---

<details>
<summary><b>🗓️ 2020/08-2021/01 開發心得：從圖靈獎大神的貼文，到壓榨極致的 9MB 模型</b></summary>
<br>
分組投入約 150 天。說到為什麼會跳下來做語音增強 (去噪音)，這一切真的只是因為在 Facebook 上看到了那一面之緣的<b>圖靈獎大神（Turing Award）發文介紹 FAIR 的最新成果</b>，腦洞大開就跟著跳坑了！

<blockquote>
  <b>🧠 底層邏輯與技術差異</b><br>
  其實，噪音去除跟「聲音分離 (Source Separation)」可以做聯想，兩者的基本概念差不多。差別在於：噪音去除是純粹把「非人聲」的頻段或特徵給過濾掉（實作時記得要注意音檔是否為多通道 Multi-channel）。
</blockquote>

<blockquote>
  <b>🧪 數據「煉丹」的眉角：算力與語系的拉扯</b><br>
  做這個項目時，我們一樣彙整了相當多的學術論文和實驗結果。深度學習都是數據為王，但「去噪」任務的數據集相對好處理很多！因為網路上到處都能找到乾淨的語音跟純噪音，只要寫程式進行動態的調整與合併（Mix），就可以無限生成數量龐大的訓練數據集。<br><br>
  <b>這時你唯一需要考量的有兩點：</b><br>
  1. 你的 GPU 記憶體（VRAM）夠不夠大，能不能把這些海量音頻特徵整個吃下來？<br>
  2. 你的乾淨人聲數據集是不是「全英文」？如果你想要擁有極佳的「中文」去噪效果，在混合數據時就必須加入大量在地的中文語料。
</blockquote>

<blockquote>
  <b>🚀 實戰成果與效能突破</b><br>
  經過無數次的架構修剪與調參，順道一提，我們最終煉出來的模型大小，是經過極致優化的 <b>9 MB</b>！而且實時比 (RTF, Real Time Factor) 高達 <b>0.08</b>。這意味著在極低的硬體資源下，也能實現極速的即時語音降噪！
</blockquote>

</details>

### 🎧 音訊超解析與音質提升 (Audio Super-Resolution & Enhancement)

* **[[NovaSR]](https://github.com/ysharma3501/NovaSR)** `[2026-04-18]` 🔥
  * **核心優勢**：**打破大模型算力迷思的 TinyML 奇蹟！僅 52KB 的微型音訊超解析 (Super-Resolution) 模型**。能透過預測並補全高頻段，將 16kHz 的「電話級」沉悶音質，瞬間升頻為 48kHz 的高保真 (Hi-Fi) 全頻帶音訊。在 A100 上高達 3600 倍實時處理速度，即使在手機 CPU 上運行也幾乎零延遲。
  * **解決痛點 / 推薦場景**：**完美解決傳統音訊增強模型動輒數百 MB，無法在邊緣設備離線運行的致命痛點**。極度輕量的體積使其能無縫嵌入 TWS 無線藍牙耳機晶片、手機 NPU 或微控制器 (MCU)。是打造 **VoIP 網路通話即時增強**、**千路直播語音端側優化**，以及 **TTS (文字轉語音) 輸出高音質後處理** 的工業級黑科技。
  * **資源**：[🐙 GitHub](https://github.com/ysharma3501/NovaSR) | [🤗 線上 DEMO 與權重](https://huggingface.co/spaces/YatharthS/NovaSR)

---

## Speaker-Separation
**👥 Speaker Separation (中文語者分離)**

> **💡 核心觀念**：從混疊的聲音訊號中提取出單一目標使用者的聲音。這是為了解決經典的**「雞尾酒會問題 (Cocktail Party Effect)」**，即在多人同時說話的吵雜場景中，精準分離出每個人獨立的聲軌。
> 👉 [🌐 更多資源：TWMAN 語者分離技術總結](https://www.twman.org/AI/ASR/SpeechSeparation) | [🤗 HF Space Demo 體驗](https://huggingface.co/spaces/DeepLearning101/Speech-Separation)


### 1. 實戰模型與應用工具 (Practical Tools)
* **[ClearVoice](https://github.com/modelscope/ClearerVoice-Studio)** `[2024-12-07]` 🔥
  * **核心優勢**：阿里開源的「一站式」語音處理大作。不僅做降噪，還能做極限的多人分離與特徵提取。[🤗 線上 Demo](https://huggingface.co/spaces/alibabasglab/ClearVoice) | [📝 中文解讀](https://juejin.cn/post/7445237715863093275)
* **[SoloSpeech](https://github.com/WangHelin1997/SoloSpeech)** `[2025-06-03]`
  * **核心優勢**：一鍵提取指定說話者的音訊，大幅提升人聲清晰度。[📝 中文解讀](https://zhuanlan.zhihu.com/p/1913305854289097038)
* **[TOLD (ICASSP 2023)](https://zhuanlan.zhihu.com/p/650346578)**：能對混疊語音直接建模的說話人日誌 (Speaker Diarization) 框架。
* **[Mossformer (ICASSP 2023)](https://zhuanlan.zhihu.com/p/609728122)**：效能極佳的單聲道語音分離模型。

### 2. 經典論文與底層架構 (Classic Papers & Architectures)
了解語音分離的底層演進，以下是必讀的學術基石：
* **[Stabilizing Label Assignment for Speech Separation](https://github.com/SungFeng-Huang/SSL-pretraining-separation)**：基於自監督預訓練 (Self-supervised Pre-training) 的分離框架。[📄 arXiv:2010.15366](https://arxiv.org/abs/2010.15366)
* **[Sudo rm -rf](https://github.com/etzinis/sudo_rm_rf)**：高效的通用音訊源分離網路。[📄 arXiv:2007.06833](https://arxiv.org/abs/2007.06833) | [🐙 Asteroid 實作](https://github.com/asteroid-team/asteroid/blob/master/asteroid/models/sudormrf.py)
* **[Dual-Path Transformer (DPTNet)](https://arxiv.org/pdf/2007.13975v3.pdf)**：直接對上下文感知的端對端單聲道語音分離。
* **[Dual-path RNN (DPRNN)](https://github.com/JusperLee/Dual-path-RNN-Pytorch)**：用於時域分離的高效長序列建模。[📄 arXiv:1910.06379](https://arxiv.org/pdf/1910.06379.pdf) | [📝 閱讀筆記](https://zhuanlan.zhihu.com/p/104606356)

---

<details>
<summary><b>🗓️ 2020/08-2021/01 開發心得：破解雞尾酒會問題的資料煉丹術</b></summary>
<br>
投入約 150 天。如同做語音常踩的坑，比較常碰到的是在做網路架構參數調整時「導致 loss 壞掉」等等問題。反而因為數據集造成的問題少很多，網路上很容易找到各種資料，比賽也多，有各種模型架構的結果可以參考（當然，多數一樣是英文數據）。

<blockquote>
  <b>🚧 語音數據合成的藝術：完全重疊 vs 部分重疊</b><br>
  語音坑最棒的地方在於，只要有了像 <code>aishell</code> 等乾淨的數據集，你想要切割或合併成「混合語音」都不是太大的問題！<br><br>
  這裡做數據相對簡單一點：我們直接把數據集打散混合，隨機挑選兩個人，然後分別挑出語音做混合 (Mix)。<b>如果長度不同，就選擇短者為參考，將長者切到與短者相同。</b><br><br>
  但要注意的是，<b>「兩兩完全重疊」與「兩兩互不完全重疊」對模型效果有不小的影響</b>！我們第一波產出的數據是兩兩完全重疊的，後來為了解決不完全重疊的現實場景，又額外產出了第二波升級版數據。
</blockquote>

<blockquote>
  <b>📊 實戰數據配方大公開</b>
  <ul>
    <li><b>第一版 (完全重疊)</b>：Train 約 5 萬多筆 (32小時) / Val 約 1 萬多筆 (10小時) / Test 約 9 千多筆 (6小時)。</li>
    <li><b>第二版 (互不完全重疊)</b>：Train 約 9 萬多筆 (112小時) / Val 約 2 萬多筆 (26.3小時) / Test 約 2 萬多筆 (29.4小時)。</li>
  </ul>
</blockquote>

<blockquote>
  <b>🧠 追隨大神腳步與架構大亂鬥</b><br>
  中間意外發現了 Google Brain 的 <code>wavesplit</code>，在有噪音及兩人同時講話情形下感覺效果不差，但沒找到相關 code，未能進一步驗證。<br><br>
  而且，又是那位有一起用餐之緣的深度學習大神 <b>Yann LeCun</b>！繼發文介紹完去噪後，又發文介紹了語音分離。後來我們陸續研究了各種架構，包含把 NLP 最早應用的 Transformer 導入的 <code>DPT-NET (Dual-path transformer)</code>、<code>DP-RNN</code>，還有 <code>VoiceFilter</code>、<code>TasNet</code>、<code>Conv-TasNet</code> 跟 <code>sudo-rm-rf</code> 等等。
</blockquote>

<blockquote>
  <b>🎓 台大李宏毅老師的指點與 CPU 落地的最後一哩路</b><br>
  這段旅程絕對不能錯過台大電機李宏毅老師的 <code>SSL-pretraining-separation</code> 論文（務必去看李老師的影片！）。最後也是多虧李老師及第一作者黃同學的解惑，小夥伴們才又更深入地確認並且解決問題。<br><br>
  在工程端，我們也深入研究了 <code>Data Parallel</code> 跟 <code>Distributed Data Parallel (DDP)</code> 的差異。但說到底，<b>如何才能在 CPU 上跑得又快又準，才是這個專案真正能落地的關鍵！</b>
</blockquote>

</details>

---

## Speech-Synthesis
**🗣️ Chinese Speech Synthesis & TTS (中文語音合成與音色克隆)**

*「想做有聲書、全自動短影音，還是專屬的虛擬 VTuber 聲優？目前的 TTS 技術不僅告別了傳統的『機器人平淡嗓音』，還能做到 3 秒極速複製你的聲音。本清單為你拆解目前最主流的歐美大廠方案與亞洲霸榜神作，讓你根據資安需求與硬體條件精準選型。」*

### 💡 效能指標：如何評估一個 TTS 模型的好壞？
* **Zero-Shot Cloning (零樣本克隆)**：模型不需要重新訓練 (Fine-tuning)，只需聽你講 3~10 秒的聲音，就能直接用你的音色唸出新稿子。
* **Latency (延遲)**：對於即時對話的 AI Agent，生成聲音的延遲至關重要。像 Orpheus TTS 標榜的 25ms 延遲，就是針對即時互動場景設計的。
* **Prosody (韻律感)**：聲音像不像真人，關鍵在於模型能否根據上下文自動加入呼吸聲、停頓和重音。

---

### 1. 歐美 AI 巨頭與國際開源社群 (資安友善 / 輕量部署篇)
如果你對專案的「原產地」有嚴格要求，或者伺服器沒有配備頂級 GPU，以下由歐美巨頭或國際社群主導的專案是你的首選：

| 模型/工具名稱 | 開發團隊/生態 | 💡 核心優勢與解決痛點 | 🚀 推薦場景與資源 |
| :--- | :--- | :--- | :--- |
| **Voxtral TTS (4B)** | 🇫🇷 **Mistral AI** `[2026-03-27]` | **34 億參數跑在手機上**。生成速度是真人說話的 6 倍，延遲 < 0.1 秒。權重全開源 MIT 協議！支援英、法、德等 9 語系 (暫無中文)。 | 筆電端本地部署、多語系<br>`[極低延遲]`<br>[🤗 Model](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) |
| **OpenAI Edge TTS** | 🇺🇸 **微軟/OpenAI 生態** | **完全免算力、免費白嫖！** 透過呼叫微軟 Edge 瀏覽器的語音介面，免 GPU 就能產出高水準語音。 | 輕量網頁應用、零成本<br>`[免 GPU]`<br>[🐙 GitHub](https://github.com/travisvn/openai-edge-tts) |
| **Parler-TTS** | 🇺🇸/🇪🇺 **Hugging Face** `[2024-09]` | **安裝最無腦的輕量之王**。HF 官方開源，主打「一行指令安裝」，對開發者極度友善。 | PoC 開發、英/歐語系<br>`[極易部署]`<br>[🐙 GitHub](https://github.com/huggingface/parler-tts) |
| **Kokoro-TTS** | 🌐 **國際開源社群** `[2024-11]` | **歐美社群熱推平替**。架構輕量且聲音自然，是取代龐大模型的優質選擇。 | 本地輕量語音助理<br>`[社群熱推]`<br>[🌐 官方介紹](https://kokorotts.net/zh-Hant) |
| **VALL-E X / DragonV2.1** | 🇺🇸 **Microsoft (微軟)** `[2025-07 更新]` | **跨語言音色保留**。微軟的經典架構與最新模型，技術底蘊深厚。 | 企業多語種配音<br>`[大廠背書]`<br>[📝 VALL-E 教學](https://mp.weixin.qq.com/s/Fo8ESzbEfjZQNUUx_giJRA) |
| **Orpheus TTS** | 🌐 **開源社群** `[2025-03]` | **即時對話王者**。25ms 超低延遲，專為即時雙向對話設計。 | 語音 AI Agent<br>`[超低延遲]`<br>[🐙 GitHub](https://github.com/canopyai/Orpheus-TTS) |
| **NeuTTS Air** | 🌐 **開源社群** `[2025-10]` | **主打端側運算 (On-Device)**。極小體積與超低功耗，可直接部署於 iOS/Android。 | 離線隱私保護 APP<br>`[端側部署]`<br>[🐙 GitHub](https://github.com/neuphonic/neutts-air) |

---

### 2. 亞洲/中國開源霸榜神作 (極致擬真 & 零樣本克隆篇)
*技術客觀評析：在「中文」表現上，以下模型目前領先全球。它們不僅精確掌握中文發音，甚至能生成帶有「笑聲、嘆氣、語氣詞」的超擬真語音。（註：注重地緣資安的專案，建議於完全離線的沙盒環境中運行）*

| 模型名稱 | 開發團隊 | 💡 核心優勢與突破點 | 🚀 推薦場景與資源 |
| :--- | :--- | :--- | :--- |
| **GPT-SoVITS** | 🇨🇳 **RVC-Boss** | **人聲克隆無冕王！** 只要 1 分鐘語音樣本就能完美複製聲音，GitHub 狂攬 35k+ Stars。 | VTuber 聲優、有聲書<br>`[霸榜神作]`<br>[🐙 GitHub](https://github.com/RVC-Boss/GPT-SoVITS) |
| **ChatTTS** | 🇨🇳 **2noise** `[2024-06]` | **打破 AI 機械音的終極武器**。支援在語句中加入「笑聲」、「停頓」，擬真度極高。 | AI Podcast、劇情對白<br>`[情緒控制]`<br>[🐙 GitHub](https://github.com/2noise/ChatTTS) |
| **Qwen3-TTS** | 🇨🇳 **阿里巴巴** `[2025-12]` | **不只克隆，還能「捏聲音」**。提供 VoiceDesign (音色創造) 與 VoiceClone (音色克隆)。 | 遊戲 NPC 配音<br>`[音色創造]`<br>[📝 中文解讀](https://zhuanlan.zhihu.com/p/1987225312841445557) |
| **Fun-CosyVoice3** | 🇨🇳 **阿里通義百聆** `[2025-12]` | **極速克隆專家**。只需短短 3 秒錄音，就能複製並轉換成 9 種不同的語言。 | 出海行銷影片翻譯<br>`[3秒克隆]`<br>[🐙 GitHub](https://github.com/FunAudioLLM/CosyVoice) |
| **MOSS-TTSD / F5-TTS** | 🇨🇳 **復旦 / 上海交大** `[2025-07]` | **學術界頂規猛獸**。MOSS 經百萬小時訓練；F5-TTS 15 秒樣本完成克隆。 | 底層架構二次開發<br>`[巨量訓練]`<br>[🌐 MOSS Project](https://www.open-moss.com/en/moss-ttsd/) |

---

### 🔥 2025-2026 前沿創新與特殊場景模型 (Special Cases)

* **[[MAGIC-TTS]](https://yongaifadian1.github.io/MAGIC-TTS/)** `[2026-04-23]` 🔥
  * **核心優勢**：**首創「毫秒級」Token 級局部時長與停頓顯式控制系統，賦予 AI 語音「導戲級」的節奏感**。華南理工大學團隊力作，打破了現代 TTS 模型僅能全局調速的「黑盒」限制。透過雙重時間建模與零值校正技術，開發者能精確指定每一個字的發音長度（如將「左」拉長 100ms）以及任意位置的停頓時長（如驗證碼中間精準停頓 260ms），且完全不影響整句的自然度。
  * **解決痛點 / 推薦場景**：**徹底解決了傳統 TTS 在「高辨識需求」場景下語速失控或節奏不明的痛點。** 完美適配於**驗證碼/訂單號播報**（分組強調）、**車載導航指令**（轉向關鍵詞重音）、**語言教學與糾錯**（精確控制讀音細節）以及**戲劇化台詞生成**。它是目前市場上少數能同時滿足「聽得自然」與「說得精準」的工業級精細化控制方案。
  * **資源**：[🐙 GitHub](https://github.com/yongaifadian1/MAGIC-TTS/) | [🤗 HuggingFace 權重](https://huggingface.co/maimai11/MAGIC-TTS) | [🌐 官方線上 Demo](https://yongaifadian1.github.io/MAGIC-TTS/)

* **[[Fish Audio S2-Pro]](https://github.com/fishaudio/fish-speech)** `[2026-03-09]` 🔥
  * **核心優勢**：**全面超越閉源系統的 TTS 新霸主，首創「自然語言內聯控制」與 Dual-AR 架構！** 這款完全開源的模型（提供權重、訓練代碼與 SGLang 推論引擎）在中英文基準測試（如 Seed-TTS Eval）中，字錯誤率（WER）強勢擊敗了 Qwen3-TTS 與 Seed-TTS 等強敵。其最大突破是支援超過 15,000 種自然語言標籤（如 `[whisper in small voice]` 或 `[professional broadcast tone]`），讓開發者無需死記固定代碼，就能精準控制情感、語氣與節奏。
  * **解決痛點 / 推薦場景**：**解決了傳統高保真 TTS 模型推論極慢、以及多說話人場景切換困難的致命痛點。** 透過 4B 慢速 AR 與 400M 快速 AR 的雙軌架構，搭配 GRPO 強化學習與 SGLang 引擎優化，單張 H200 即可達成 0.195 的極低即時因子（RTF）與約 100ms 的首音頻延遲。極度適合打造 10~30 秒極速零樣本音色克隆、單次生成多說話人的沈浸式廣播劇，以及高併發的即時多輪對話智能體。
  * **資源**：[🐙 GitHub](https://github.com/fishaudio/fish-speech) | [🤗 HuggingFace 權重](https://huggingface.co/fishaudio/s2-pro) | [🌐 官方網站](https://fish.audio/) | [📄 技術報告與部落格](https://fish.audio/blog/fish-audio-open-sources-s2/)

  <details>
  <summary><b>📊 核心實測對標數據 & 💻 生產環境部署指南 (點擊展開)</b></summary>

  #### 🏆 基準測試壓制性表現
  * **Seed-TTS Eval 中文 WER**：0.54%（所有對比模型中最低，優於 MiniMax Speech-02 的 0.99% 與 Qwen3-TTS 的 0.77%）。
  * **Seed-TTS Eval 英文 WER**：0.99%（所有對比模型中最低，優於 Seed-TTS 的 2.25%）。
  * **Audio Turing Test**：得分高達 0.515（顯著超越 Seed-TTS 的 0.417）。
  * **EmergentTTS-Eval**：以 81.88% 取得最高得分。

  #### 🚀 SGLang 生產推論效能 (單張 H200 GPU)
  * **極致優化**：支援連續批次處理、分頁 KV 快取、CUDA 圖重放與 RadixAttention 前綴快取。
  * **高吞吐量**：突破 3,000+ tokens/s。
  * **自動快取 KV 狀態**：在聲音克隆場景中，同一聲音重複使用時前綴快取命中率平均達 86.4%（峰值 >90%），大幅降低算力開銷。

  #### 💻 開箱即用部署指令
  **環境安裝與 CLI 推論：**
  ```bash
  git clone [https://github.com/fishaudio/fish-speech.git](https://github.com/fishaudio/fish-speech.git)
  cd fish-speech
  pip install uv
  uv sync

  python -m fish_speech.text_to_speech \
  --text "你好，我是 Fish Audio S2-Pro" \
  --reference_audio reference.wav \
  --output output.wav
  Docker 極速部署：

  Bash
  docker pull fishaudio/fish-speech:latest
  docker run -it --gpus all fishaudio/fish-speech:latest
  (生產環境推薦搭配 SGLang 伺服器：sglang-omni)
  </details>


* **[[Xiaomi Any2Speech]](https://Any2Speech.github.io/)** `[2026-04]` 🔥
  * **核心優勢**：**定義「導戲級」控制力，首創 GST 分層協議實現長音頻情緒弧線。** 小米大模型團隊力作，將 TTS 從單句朗讀升級為「舞台級」演繹。透過 Global-Sentence-Token 三層架構與 CoT 思維鏈，能理解劇本邏輯並自動規劃長達 10 分鐘的情緒起伏、說話意圖與聲學場景，讓人聲、環境音與背景細節完美融合。
  * **解決痛點 / 推薦場景**：**徹底解決長篇音頻（如廣播劇、武俠評書）中角色狀態不連貫、環境音與人聲分離感重的「錄音棚感」。** 支援多角色複雜對白與長文自動轉譯。是打造沈浸式 AI 廣播劇、長篇播客自動生產、以及需要細膩情感遞進（如辯論、脫口秀）場景的工業級首選。
  * **資源**：[🌐 專案主頁](https://Any2Speech.github.io/) | [⚙️ OpenClaw 技能](https://clawhub.ai/whiteshirt0429/xiaomi-Xiaomi%20Any2Speech-beyondtts)

* **[[Midasheng-audio-generate]](https://nieeim.github.io/Dasheng-AudioGen-Web/)** `[2026-04]` ✨
  * **核心優勢**：**「一句話造世界」，首個全能型沈浸式全場景音頻生成框架。** 採用 Flow Matching 框架與 Midasheng Tokenizer，實現人聲、音樂、環境音的「一體化生成」。只需自然語言描述（如「雨中偵探獨白」），模型即可自動配置對應的混響、背景氛圍與情緒語音，無需任何後期拼接。
  * **解決痛點 / 推薦場景**：**解決了音訊後期製作中，人聲與環境音混響不匹配、音效素材尋找困難的痛點。** 支援細粒度分層控制與智能體（Agent）工作流。適合快速生成具備高度場景感的短影音配音、遊戲 NPC 環境語音，以及需要「聲畫同頻」的沈浸式內容創作。
  * **資源**：[🎵 Demo 頁面](https://nieeim.github.io/Dasheng-AudioGen-Web/) | [⚙️ OpenClaw 技能](https://clawhub.ai/jimbozhang/midasheng-audio-generate)

* **[[Woosh]](https://github.com/SonyResearch/Woosh)** `[持續更新]` 🔥
  * **核心優勢**：**索尼 (Sony Research) 釋出的超級大禮，工業級 AI 動態音效 (Foley) 生成神器！** 補足了生成式 AI 在「擬音與環境音效」領域的空白。承襲索尼影業與 PlayStation 的深厚聲學底蘊，它生成的音效不再是粗糙的白噪音拼接，而是具備極強空間感與物理真實感的高保真音頻 (如物體劃破空氣的呼嘯聲、魔法技能音效)。
  * **解決痛點 / 推薦場景**：**徹底解決短影音後期、獨立遊戲 (Indie Games) 開發與廣播劇製作時「找不到合適音效」與「高昂素材版權費」的致命痛點。** 帶動了音效製作從「素材庫檢索」到「按需客製化生成」的典範轉移。只要輸入文本提示 (Prompt，如「跑車飛馳而過」)，就能讓缺乏專業錄音棚的零預算小團隊，輕鬆量產源源不絕的 3A 級聲音素材。
  * **資源**：[🐙 GitHub](https://github.com/SonyResearch/Woosh) | [📦 第三方一鍵整合包 (免代碼版)](https://pan.quark.cn/s/8228a6fd7384?pwd=1QxE)

* **[[OmniVoice]](https://github.com/k2-fsa/OmniVoice)** `[2026-04-02]` 🔥
      * **核心優勢**：**0.8B 極小參數達成 600+ 語言覆蓋，TTS 從「拚規模」轉向「拚效率」的劃時代作。** 小米 AI 實驗室（Daniel Povey 團隊）出品，不僅支援 12 種中國方言（如四川話、東北話），更實現了「零樣本音色克隆」與「情緒同步」的雙重突破。其極致輕量的架構，讓開發者能以極低硬體負擔實現「真人口吻」的細膩表達。
      * **解決痛點 / 推薦場景**：**徹底解決傳統模型因體積龐大難以落地邊緣設備，以及中英混說、方言朗讀具備「機械感」的通病。** 支援耳語、ASMR 與細粒度情緒控制，完美契合「短影音在地化旁白」、「車載語音助理」與「跨國產品出海」等場景。它將聲音轉化為「可複用數位資產」，讓中小團隊也能低門檻打造具備品牌 IP 感的虛擬員工聲音。
      * **資源**：[🐙 GitHub](https://github.com/k2-fsa/OmniVoice) | [📄 論文：OmniVoice (arXiv)](https://www.google.com/search?q=https://arxiv.org/abs/2604.XXXXX) | [🌐 官方技術解讀](https://uy6npdpeoi.feishu.cn/docx/EAWYdWWO7ormNPxUhyVcO3GSnUc)

- **[2026-02-22] Ming-flash-omni-2.0**
  - **說明**：透過簡單指令即可控制產生音訊的語速、音量、音調。
  - **資源**：[🐙 GitHub](https://github.com/inclusionAI/Ming-omni-tts) | [🤗 HF Model](https://huggingface.co/inclusionAI/Ming-flash-omni-2.0) | [📝 公眾號解讀](https://mp.weixin.qq.com/s/x3DPVL92NhO4ENm6WId-uw)

- **[2026-01-26] Chroma 1.0 (全雙工對話專用)**
  - **說明**：專為全雙工（Full-duplex）即時語音對話設計。150ms 超低延遲與隨時可打斷的特性，是開發虛擬陪伴或即時客服 Agent 的完美引擎。
  - **資源**：[🐙 GitHub](https://github.com/FlashLabs-AI-Corp/FlashLabs-Chroma) | [🤗 HF Model](https://huggingface.co/FlashLabs/Chroma-4B) | [📝 公眾號解讀](https://mp.weixin.qq.com/s/V9xctkJYAuoURqbREXHidQ)

* **[[VoxCPM 2]](https://github.com/OpenBMB/VoxCPM)** `[2026-04]` 🔥
  * **核心優勢**：**雲端聽覺藝術畫筆！OpenBMB 釋出的 2B 參數無分詞器 (Tokenizer-Free) 端到端 TTS 大模型。** 基於擴散自回歸架構，支援 30+ 語言與多種方言，能直接從 16kHz 參考音頻輸出 48kHz 高保真音訊。其「零樣本音色克隆」技術能精準模擬呼吸、語調等極致情緒細節。
  * **解決痛點 / 推薦場景**：**用算力換取擬真度上限，解決傳統 TTS 缺乏情感與自然度的痛點。** 配合 Nano-vLLM 引擎可在單張 RTX 4090 上實現低實時因子 (RTF)。是打造高階數位人/虛擬陪伴、有聲書與短影音情感配音，以及建構頂級雲端 TTS API 服務的最佳開源底座。
  * **資源**：[🐙 GitHub](https://github.com/OpenBMB/VoxCPM)
  - **[2025-12-12] VoxCPM 1.5**
    - **說明**：告別機械音的「最強嘴替」。
    - **資源**：[🤗 HuggingFace](https://huggingface.co/openbmb/VoxCPM1.5) | [📝 公眾號解讀](https://mp.weixin.qq.com/s/kmA6aZmCIhv1x0qS1kPCNg)

- **[2025-08-15] ZipVoice (純 CPU 推理)**
  - **說明**：擺脫對昂貴 GPU 的依賴，純靠 CPU 就能實現流暢語音合成。預算有限或輕量雲端伺服器部署的絕佳方案。
  - **資源**：[🐙 GitHub](https://github.com/k2-fsa/ZipVoice)

- **[2025-08-08] KittenTTS**
  - **說明**：超迷你 TTS 模型（< 25 MB）。[🐙 GitHub](https://github.com/KittenML/KittenTTS) | [📝 Reddit 討論](https://www.reddit.com/r/LocalLLaMA/comments/1mhyzp7/kitten_tts_sota_supertiny_tts_model_less_than_25/?tl=zh-hant)

- **[2025-07-25] Higgs Audio V2**
  - **說明**：李沐團隊開源，支援越南語。[🐙 GitHub](https://github.com/boson-ai/higgs-audio) | [📝 李沐教學](https://zhuanlan.zhihu.com/p/1931365847840069074)

- **[2025-07-23] FreeAudio**
  - **說明**：90秒長時可控音效生成 (如狼嚎、蟋蟀聲)。[🌐 Project](https://freeaudio.github.io/FreeAudio/)

- **[2025-06-05] OpenAudio S1**
  - **說明**：高可控多語言 TTS。[🤗 HuggingFace](https://huggingface.co/fishaudio/openaudio-s1-mini)

- **[2025-03-30] MegaTTS3**
  - **說明**：字節跳動開源 0.45B 中英雙語模型。[🤗 Demo](https://huggingface.co/spaces/ByteDance/MegaTTS3)

- **[2025-03-15] CSM (Conversational Speech)**
  - **說明**：1B 參數實現電影級人聲。[🐙 GitHub](https://github.com/SesameAILabs/csm)

- **[2025-03-01] Step-Audio**
  - **說明**：結合 ComfyUI 的聲音複製技術。[🐙 GitHub](https://github.com/stepfun-ai/Step-Audio)

- **[2024-11-30] MockingBird (經典)**
  - **說明**：5秒速「復刻」聲音，35.4k Star 神作。[🐙 GitHub](https://github.com/babysor/MockingBird)

- **[2024-??] MeloTTS**
  - **說明**：無 GPU 也可靈活使用。[🐙 GitHub](https://github.com/myshell-ai/MeloTTS)

---

## 💾 開源語音資料集 (Speech Datasets)

*沒有百萬小時的煉丹爐，生不出好模型！對於需要訓練在地化模型的開發者來說，高品質、標註乾淨的語料庫是無價之寶。以下收錄 2024-2025 釋出的重量級資料集。*

* **[語音識別資料匯總：常見庫和特徵對比](https://zhuanlan.zhihu.com/p/616020595)**：盤點各類 ASR 開源資料集的適用場景與特徵。
* **[2024-2025 開源語音資料彙整](https://zhuanlan.zhihu.com/p/1974579913194501708)**：最新年度的高品質語音與音訊訓練資料大集合。

### 🇨🇳 中文與方言 / 區域性資料集
* **[Mozilla Common Voice Datasets (zh-TW 繁體中文)](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/zh-TW)**：開源界最珍貴的「台灣口音繁體中文」語音資料集，是訓練在地化 ASR 的必備彈藥庫。
* **[WenetSpeech-Yue (2025)](https://github.com/ASLP-lab/WenetSpeech-Yue)**：21,800 小時全球最大粵語語音資料集，涵蓋 10 大領域，支援中英混雜場景。
* **[WenetSpeech-Chuan](https://github.com/ASLP-lab/WenetSpeech-Chuan)**：10,000 小時首個大規模川渝方言語料庫。
* **[Easy-Turn-Trainset](https://www.modelscope.cn/datasets/ASLP-lab/Easy-Turn-Trainset)**：約 1100 小時對話輪次檢測資料集，用於全雙工對話系統（包含完整、回應、等待狀態）。
* **[Chinese-LiPS (2025-05)](https://kiri0824.github.io/Chinese-LiPS/)**：100 小時中文「多模態」語音辨識資料集。首創結合「唇讀資訊 + 投影片語意」，多模態融合後 CER 降至 2.58%。
* **[CS-Dialogue (2025-02)](https://huggingface.co/datasets/BAAI/CS-Dialogue)**：104 小時目前最大的公開自發式「中英切換」對話資料集，捕捉真實自然的語言切換現象。
* **特殊人群資料集 (智源研究院, 2025-04)**：
  * **[ChildMandarin](https://huggingface.co/datasets/BAAI/ChildMandarin)**：填補低幼兒童 (41.25 小時) 語音數據空白。
  * **[SeniorTalk](https://huggingface.co/datasets/BAAI/SeniorTalk)**：世界首個中文超高齡老人 (55.53 小時) 對話資料集。
* **[Emilia (2024-08)](https://huggingface.co/datasets/Amphion/Emilia)**：101,000 小時目前最大多語種語音生成資料集，支援中英等 6 語系，涵蓋脫口秀與辯論場景。
* **[GigaSpeech 2 (2024-06)](https://huggingface.co/datasets/speechcolab/gigaspeech2)**：30,000 小時東南亞多語言（泰語、印尼語、越南語）資料集，模型效能達商業水準。
* **[LLaSO (2024)](https://github.com/EIT-NLP/LLaSO)**：開源語音大模型框架，包含 1200 萬對齊樣本與 1350 萬多任務指令樣本。

### 🌍 國際與醫療/情緒等特殊應用資料集
* **[Meta Omnilingual ASR Corpus (2025-11)](https://github.com/facebookresearch/omnilingual-asr)**：支援 1600+ 種語言的大規模轉錄資料集，具備少樣本學習能力，可擴展至 5400+ 語言。[🌐 線上示範](https://aidemos.atmeta.com/omnilingualasr/language-globe)
* **[HiFiTTS-2 (2025)](https://huggingface.co/datasets/nvidia/hifitts-2)**：專注高頻寬 (22.05kHz / 44.1kHz) 語音合成的英語資料集，規模達數萬小時，支援零樣本 TTS 訓練。
* **[Common Voice (v22.0, 2025-06)](https://datacollective.mozillafoundation.org/datasets)**：全球最大眾包語音資料集，累積近 10 萬人參與，錄製達 3,718 小時，覆蓋 137 種語言。
* **[Bridge2AI-Voice (2025-01)](https://physionet.org/content/b2ai-voice/1.1/)**：由 NIH 推進的醫療語音資料集，針對語音障礙、神經系統疾病及憂鬱症等進行收音 (需申請)。
* **[VietMed](https://github.com/leduckhai/multimed)**：越南醫療語音資料集，涵蓋所有 ICD-10 疾病組及當地口音。
* **[nEMO (2024-04)](https://huggingface.co/datasets/amu-cai/nEMO)**：3 小時波蘭語「情緒」語音資料集，包含憤怒、恐懼、快樂等 6 種極端情緒。

---

## Speech-Applications
**🎙️ 語音處理綜合應用 (All-in-One Applications)**

當單項技術（辨識、合成、分離）趨於成熟，2026 年的趨勢在於將多個模型串聯成解決特定生活痛點的「完整方案」。以下收錄具備高度整合性且支援本地部署的開源神作：

### 🎼 音樂與歌聲處理 (Music & Singing)

* **[[Nightingale]](https://github.com/rzru/nightingale)** `[2026-04-18]` 🔥
  * **核心優勢**：**頂尖本地開源卡拉OK系統，整合 ASR 與音源分離技術的家庭 KTV 終極方案**。採用 Rust 核心開發，完整封裝了 `UVR Karaoke` (音軌分離)、`WhisperX` (詞級時間軸對齊) 與 `Pitch Scoring` (音調打分) 引擎。支援 Windows/macOS/Linux 全平台硬體加速（CUDA/MPS），能將任意本地 MP3/影片自動轉化為帶有同步歌詞、動態背景且可導唱的 K歌房等級體驗。
  * **解決痛點 / 推薦場景**：**完美解決「冷門歌曲無伴奏、無動態歌詞」與「雲端軟體收費/版權限制」的痛點**。適合擁有大量無損音樂收藏的燒友，在客廳搭建完全隱私、免設定、支援手把操作的「純本地 AI 點歌台」。
  * **資源**：[🐙 GitHub](https://github.com/rzru/nightingale) | [📄 WhisperX 引擎](https://github.com/m-bain/whisperX) | [📝 核心架構解析](https://github.com/rzru/nightingale#technical-details)

* **[[AudioX]](https://github.com/ZeyueT/AudioX-ICLR)** `[2026-03-15]` 🔥
  * **核心優勢**：**打破模型碎片化！單一模型實現「任意模態到音訊」的終極生成架構 (ICLR 2026 接收)**。由港科大團隊開發，基於 DiT (Diffusion Transformer) 與多模態自適應融合模組 (MAF)，**一個模型就能同時包辦文字生音效 (T2A)、文字生音樂 (T2M)、影片自動配音 (V2A) 以及音訊修復與續寫**。
  * **解決痛點 / 推薦場景**：**完美解決過去影片配音與音效生成需要串接多個專用模型，且無法精準控制聲音發生時間軸的痛點**。支援細粒度的時間戳控制（例如：指定 1.6 秒到 4.4 秒出現沖水聲）與多樂器風格指定。是遊戲音效設計、自動化影視配樂、以及短影片動態配音的工業級全能底座。團隊更同步開源了高達 700 萬樣本的高品質標註資料集 IF-caps。
  * **資源**：[🐙 GitHub](https://github.com/ZeyueT/AudioX-ICLR) | [📄 論文](https://arxiv.org/pdf/2503.10522) | [🤗 線上 DEMO](https://huggingface.co/spaces/HKUSTAudio/AudioX-Demo) | [📊 IF-caps 資料集](https://huggingface.co/datasets/HKUSTAudio/IF-caps)

### 🗣️ 會議轉寫與多語者分離 (Meeting Transcription & Diarization)

* **[[Whisper + CAM++ 離線轉寫管線]](https://modelscope.cn/models/iic/speech_campplus_speaker-diarization_common)** `[2026-04-18]` 🔥
  * **核心優勢**：**完美互補！將 Whisper 的超強語音辨識與阿里 CAM++ 的精準聲紋分離結合，打造零成本的純本地會議轉寫神器**。Whisper 本身無法區分說話人，透過導入 CAM++ 進行語音活動偵測 (VAD) 與聲紋聚類 (Speaker Clustering)，能精準自動標註「發言人1、發言人2」，徹底彌補了單一開源 ASR 模型無法辨識語者身分的缺陷。
  * **解決痛點 / 推薦場景**：**解決了企業機密會議無法上傳雲端，以及長音訊（數小時）人工聽打極度耗時的痛點**。支援純 CPU 運行且全離線部署，確保資料絕對隱私。是企業內部會議紀錄自動化、自媒體/播客 (Podcast) 自動生成多語者字幕，以及高機密訪談整理的工業級落地首選。
  * **資源**：[🐙 CAM++ 模型與技術文件](https://modelscope.cn/models/iic/speech_campplus_speaker-diarization_common) | [📄 Whisper 論文](https://cdn.openai.com/papers/whisper.pdf) | [📝 實戰落地指南與程式碼](https://mp.weixin.qq.com/s/Kkzkcs85_kYTWMRQnpOXlA)



<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "mainEntityOfPage": {
    "@type": "WebPage",
    "@id": "https://deep-learning-101.github.io/Speech-Processing"
  },
  "headline": "2026 語音處理 (Speech Processing) 資源與模型大全",
  "description": "一份詳盡的語音處理（Speech Processing）資源清單，涵蓋語音識別(ASR)、語者識別、語音增強、語者分離與語音合成(TTS)等領域的最新研究與開源工具，解決Podcast逐字稿、邊緣運算與即時語音對話痛點。",
  "image": "https://raw.githubusercontent.com/Deep-Learning-101/TonTon/refs/heads/main/_includes/DL101-Logo.jpg",
  "author": {
    "@type": "Organization",
    "name": "Deep Learning 101, Taiwan",
    "url": "https://deep-learning-101.github.io/"
  },
  "publisher": {
    "@type": "Organization",
    "name": "Deep Learning 101, Taiwan",
    "logo": {
      "@type": "ImageObject",
      "url": "https://raw.githubusercontent.com/Deep-Learning-101/TonTon/refs/heads/main/_includes/DL101-Logo.jpg"
    }
  },
  "datePublished": "2026-03-29",
  "dateModified": "2026-03-29",
  "keywords": "語音處理, Speech Processing, ASR, TTS, 語音辨識, 語音合成, 語者分離, 聲音克隆, Whisper, 本地部署, 逐字稿生成, AI客服, 邊緣運算",
  "about": {
    "@type": "Service",
    "serviceType": "AI Consulting",
    "provider": {
      "@type": "Organization",
      "name": "Deep Learning 101, Taiwan"
    },
    "name": "人工智慧顧問服務 (AI Consulting)",
    "description": "提供關於語音處理（Speech Processing）領域的專業顧問服務，包含語音識別（ASR）、語音合成（TTS）、模型開發與技術導入。"
  }
}
</script>