#
https://www.twman.org/AI/ASR

https://huggingface.co/DeepLearning101

https://deep-learning-101.github.io/Speech-Processing

https://github.com/Deep-Learning-101/Speech-Processing-Paper
#

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

## Speech-Processing
**語音處理 (Speech Processing)**

- 2025-09-23｜**Sherpa onnx**
  - 資源：[🐙 GitHub](https://github.com/k2-fsa/sherpa-onnx)

- 2025-05-14｜**TEN Framework**
  - 資源：[📝 ten-turn-detection](https://zread.ai/TEN-framework/ten-turn-detection) | [📝 ten-vad](https://zread.ai/TEN-framework/ten-vad)

- 2025-01-19｜**觀點文章**
  - 標題：[小米語音首席科學家 Daniel Povey：語音辨識捲完了，下一個機會在哪裡？](https://www.jiqizhixin.com/articles/2025-01-19-4?)

- **踩坑指南 (必讀)**
  - [ASR/TTS 開發避坑指南](https://blog.twman.org/2024/02/asr-tts.html) (強調數據質量)
  - [那些語音處理踩的坑](https://blog.twman.org/2021/04/ASR.html) (實務經驗分享)

- **基礎知識 & 資料集**
  - [音視頻開發基礎入門 (聲音採集、量化、碼率)](https://zhuanlan.zhihu.com/p/577850804)
  - [萬字語音合成基礎與論文總結](https://mp.weixin.qq.com/s/S9T9fk9THUF3JQRnNuOM7Q)
  - [Mozilla Common Voice Datasets - zhTW](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/zh-TW)
  - [語音識別資料匯總：常見庫和特徵對比](https://zhuanlan.zhihu.com/p/616020595)
  - [2024-2025 開源語音資料彙整](https://zhuanlan.zhihu.com/p/1974579913194501708)

---

## Speech-Recognition
**中文語音識別 (Chinese Speech Recognition)**
> 通過語音信號處理和模式識別讓機器自動識別和理解人類的口述。
> [🌐 更多 ASR 資源](https://www.twman.org/AI/ASR)

-----

### 👑 2026 全球開源 ASR 語音辨識模型大比拚 (非中/歐美大廠篇)

**1. Whisper 生態系擴充：效能與速度的極致壓榨**

| 模型/工具名稱 | 開發源頭/生態 | 💡 解決什麼痛點？ (核心優勢) | 🚀 推薦適用場景 & 規格標籤 |
| :--- | :--- | :--- | :--- |
| **WhisperX** | 歐美開源社群 | **精準時間戳與語者辨識**：原版 Whisper 常常把不同人的對話糊在一起，它能強力對齊字級時間戳。 | 適合：會議紀錄、影片自動上字幕<br>`[需 GPU]` `[高精度時間戳]` |
| **Distil-Whisper** | 歐美開源社群 | **輕量化與極速**：模型縮小 49%，速度提升 6 倍，但保留了 99% 的辨識精準度。 | 適合：算力有限的本地伺服器<br>`[低顯存需求]` `[英文效能極佳]` |
| **Insanely-Fast-Whisper** | 歐美開源社群 | **天下武功唯快不破**：透過底層優化，讓 Whisper 的推理速度達到令人髮指的地步。 | 適合：需要批次處理海量音檔的企業<br>`[極速轉寫]` `[吞吐量王者]` |
| **CarelessWhisper** | 歐美開源社群 | **低延遲串流辨識**：微調 Whisper 實現接近非串流式的精準度，適合即時應用。 | 適合：即時語音助理、直播實時字幕<br>`[低延遲]` `[即時辨識]` |

**2. 歐美 AI 巨頭的逆襲：次世代 ASR 模型**

| 模型名稱 | 開發團隊 | 💡 核心技術與亮點 | 🚀 推薦適用場景 & 規格標籤 |
| :--- | :--- | :--- | :--- |
| **Parakeet-tdt-0.6b-v3** | 🇺🇸 **NVIDIA** (輝達) | **1秒轉錄1小時音訊！** 輝達推出的最強開源模型之一，吞吐量極其驚人。 | 適合：具備高等級 GPU 算力的企業級資料清洗<br>`[NVIDIA 生態]` `[極限速度]` |
| **Voxtral (Small 24B/Mini 3B)** | 🇫🇷 **Mistral AI** | **超越 GPT-4o mini 的語音能力**，歐洲 AI 巨頭的首個開源語音模型。 | 適合：需要整合大型語言模型的語音應用<br>`[歐美頂規]` `[多語種]` |
| **OpusLM** | 🇺🇸 **CMU** (卡內基梅隆) | 學術界重磅！統一了語音辨識、合成與文字理解的大模型。 | 適合：AI 研究人員、多模態系統開發<br>`[學術開源]` `[多模態]` |
| **MedASR** | 🇺🇸 **Google** | 專攻醫療領域的語音辨識模型，解決專業術語難以辨識的痛點。 | 適合：醫療院所、數位健康領域的語音病歷輸入<br>`[醫療專精]` `[高準確度]` |

-----

### 🌏 亞洲頂尖開源 ASR 模型 (中文語境特化篇)

*「如果你處理的音訊包含大量複雜的中文方言、中英夾雜，或是極具挑戰性的長時段錄音，以下這些由亞洲/中國科技大廠開源的模型，在中文語境的基準測試中目前處於領先地位。**（註：注重地緣資安合規的專案，請自行評估導入風險）**」*

| 模型名稱 | 開發團隊 | 💡 核心優勢與突破點 | 🚀 推薦適用場景 & 規格標籤 |
| :--- | :--- | :--- | :--- |
| **FireRedASR2S** | 🇨🇳 **小紅書** (FireRedTeam) | **SOTA 級別的工業全能系統**，第二代架構在極端場景下的辨識率非常強悍。 | 適合：短影音平台、社交媒體內容監控<br>`[中文 SOTA]` `[工業級]` |
| **Qwen3-ASR** | 🇨🇳 **阿里巴巴** | **吊打原生 Whisper**，支援高達 52 種語言和方言，阿里體系的最新力作。 | 適合：出海企業、多語種客服系統<br>`[方言支援]` `[多語種]` |
| **VibeVoice-ASR** | 🇨🇳 **開源社群** | **拒絕切片！一次吞下 60 分鐘音訊**。透過 64K 超長上下文窗口，直接吐出結構化結果。 | 適合：長篇演講、一小時以上的完整會議錄音<br>`[超長上下文]` `[免切片]` |
| **Fun-ASR** | 🇨🇳 **阿里達摩院** | **小參數大能量**，0.8B 的模型效能直逼 12B 的巨頭，且支援離線轉寫 SDK。 | 適合：邊緣運算設備、本地端低資源部署<br>`[輕量化]` `[高性價比]` |

-----

### 🔥 最新模型 (2026)

- 2026-02-25 | **FireRedASR2S**
  - 說明：目前中文開源界的 SOTA 霸主。針對短影音、直播與社交媒體中常見的複雜口音、中英夾雜與背景噪音干擾進行了深度優化，是打造高併發內容審核平台與全自動影片上字幕系統的工業級首選。
  - 資源：[🐙 GitHub](https://github.com/FireRedTeam/FireRedASR2S) | [📝 公眾號解讀](https://mp.weixin.qq.com/s/d1vYXNegQdqph_nFDDye9A)

- 2026-01-30 | **VibeVoice-ASR**
  - 說明：徹底解決傳統 ASR 模型因音檔切片導致的語意斷層與時間戳記偏移問題。支援 64K 超長上下文，能一次處理 60 分鐘音檔，極度適合一小時以上的長篇 Podcast 轉錄、企業法說會或學術演講的完整逐字稿生成。
  - 資源：[🐙 GitHub](https://github.com/microsoft/VibeVoice) | [📝 公眾號解讀](https://mp.weixin.qq.com/s/W8VVkg2igydIZgMkqw9wBA)

- 2026-01-30 | **Qwen3-ASR**
  - 說明：支援52 種語言和方言，吊打Whisper
  - 資源：[🤗 HuggingFace](https://huggingface.co/spaces/Qwen/Qwen3-ASR) | [📝 公眾號解讀](https://mp.weixin.qq.com/s/-7gm2BstDVxTkJ6lD3Znmg)

- 2025-12-23｜**MedASR**
  - 說明：Google 發布醫學語音辨識模型
  - 資源：[🤗 HuggingFace](https://huggingface.co/google/medasr)

- 2025-12-16｜**Fun-ASR**
  - 說明：主打極致性價比與輕量化，僅需極低顯存（VRAM）即可在本地端流暢運行 0.8B 模型。內建完善的離線轉寫 SDK，非常適合邊緣運算設備（Edge AI）或需在封閉內網環境部署的企業語音客服系統。
  - 資源：[🐙 GitHub](https://github.com/FunAudioLLM/Fun-ASR) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1984310683358217029) | [📝 公眾號解讀，用Fun-ASR-Nano微調一個「聽懂行話」的語音模型](https://mp.weixin.qq.com/s/M1vGqFZV5MWREkSyx2-ITw)

- 2025-12-15｜**GLM-ASR**
  - 說明：解決複雜聲學環境、方言辨識以及低音量語音
  - 資源：[🐙 GitHub](https://github.com/zai-org/GLM-ASR) | [🤗 HuggingFace](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1983951645055419349)

- 2025-11-15｜**Omnilingual-ASR**
  - 資源：[🐙 GitHub](https://github.com/facebookresearch/omnilingual-asr) | [🌐 DEMO](https://aidemos.atmeta.com/omnilingualasr/language-globe)

- 2025-08-29｜**WhisperLiveKit**
  - 說明：讓即時語音轉寫絲滑得不像話的神器
  - 資源：[🐙 GitHub](http://github.com/QuentinFuxa/WhisperLiveKit) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1944712252512010607)

- 2025-08-27｜**CarelessWhisper**
  - 說明：微調Whisper實現低延遲串流識別，效果接近非串流式
  - 資源：[🐙 GitHub](https://github.com/tomer9080/CarelessWhisper-streaming) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1977136140139141051)

- 2025-08-10｜**Canary-1b-v2**
  - 說明：NVIDIA 發布多語種語音 AI 開放資料集與模型
  - 資源：[🤗 HuggingFace](https://huggingface.co/nvidia/canary-1b-v2) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1952436345222993067)

- 2025-08-08｜**Parakeet-tdt-0.6b-v3**
  - 說明：1秒轉錄1小時音訊！輝達最強開源模型
  - 資源：[🤗 HuggingFace](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | [📝 媒體報導](https://hk.finance.yahoo.com/news/1%E7%A7%92%E8%BD%89%E9%8C%841%E5%B0%8F%E6%99%82%E9%9F%B3%E8%A8%8A-%E8%BC%9D%E9%81%94%E9%87%8D%E7%A3%85%E9%96%8B%E6%BA%90%E8%AA%9E%E9%9F%B3%E8%AD%98%E5%88%A5%E6%9C%80%E5%BC%B7%E6%A8%A1%E5%9E%8Bparakeet-075846970.html)

- 2025-07-16｜**Voxtral (Mistral)**
  - 說明：Mistral 首個開源語音模型，超越 GPT-4o mini
  - 資源：[Small 24B](https://huggingface.co/mistralai/Voxtral-Small-24B-2507) | [Mini 3B](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1928945056955471125)

- 2025-07-02｜**OpusLM**
  - 說明：CMU 發布統一語音辨識、合成、文字理解的大模型
  - 資源：[🤗 HuggingFace](https://huggingface.co/espnet/OpusLM_7B_Anneal) | [📝 中文解讀](https://mp.weixin.qq.com/s/XCgBTgfOs8y_fFFEEMrW-w)

- 2025-06-06｜**speakr**
  - 說明：開源轉錄工具，支援 AI 生成內容
  - 資源：[🐙 GitHub](https://github.com/murtaza-nasir/speakr) | [📝 中文解讀](https://cloud.tencent.com/developer/news/2645205)

- 2025-05-06｜**VITA-Audio**
  - 說明：快速交錯跨模態令牌生成
  - 資源：[📚 DeepWiki](https://deepwiki.com/VITA-MLLM/VITA-Audio) | [📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/2505.03739)

- 2025-04-28｜**FireRedASR**
  - 說明：AI 語音助理語音轉文字 API
  - 資源：[🐙 GitHub](https://github.com/FireRedTeam/FireRedASR) | [📝 教學](https://mp.weixin.qq.com/s/FUC-rSkItxEQJIWUbU4Cpw)

- 2025-04-02｜**Dolphin**
  - 說明：Large-Scale ASR Model for Eastern Languages
  - 資源：[🐙 GitHub](https://github.com/DataoceanAI/Dolphin) | [📄 arXiv](https://arxiv.org/abs/2503.20212)

- 2024-07-03｜**SenseVoice**
  - 說明：阿里開源，支援偵測掌聲、笑聲
  - 資源：[🌐 Project](https://funaudiollm.github.io/) | [📝 中文解讀](https://mp.weixin.qq.com/s/q-DyyAQikz8nSNm6qMwZKQ)

### 經典模型庫 (Classic Toolkits)

- **Whisper Family (OpenAI)**
  - [**Whisper**](https://openai.com/research/whisper): OpenAI 開源準確率最高的通用模型。
  - [**WhisperLive**](https://github.com/collabora/WhisperLive): 免費即時語音轉文字工具。
  - [**Distil-Whisper**](https://github.com/huggingface/distil-whisper): 輕量級 AI 的強大力量。
  - [**Insanely-Fast-Whisper**](https://github.com/Vaibhavs10/insanely-fast-whisper): 超快速辨識腳本。
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
<summary>2020/03-2021/01 開發心得</summary>
語音辨識（speech recognition）技術，也被稱為自動語音辨識（英語：Automatic Speech Recognition, ASR）、電腦語音識別（英語：Computer Speech Recognition）或是語音轉文字識別（英語：Speech To Text, STT），其目標是以電腦自動將人類的語音內容轉換為相應的文字；跟小夥伴們一起嘗試過NEMO還有Kaldi、MASR、VOSK，wav2vec以及Google、Azure等API，更別說後來陸續又出現SpeechBrain、出門問問的WeNet跟騰訊PIKA等。目前已知可訓練聲學模型(AM)中文語音(中國發音/用語，可惜還沒臺灣較靠譜的)公開數據如：Magic-Data_Mandarin-Chinese-Read-Speech-Corpus、aidatatang、aishell-1 、aishell-2等約2000多小時(aishell目前已到4，但想商用至少得破萬小時較靠譜)；再搭配語言模型(LM)，然後基於各種演算法架構優化各有優缺點，效果也各有優劣。與說話人辨識及說話人確認不同，後者嘗試辨識或確認發出語音的說話人而非其中所包含的詞彙內容。 語音辨識技術的應用包括語音撥號、語音導航、室內裝置控制、語音文件檢索、簡單的聽寫資料錄入等。語音辨識技術與其他自然語言處理技術如機器翻譯及語音合成技術相結合，可以構建出更加複雜的應用，例如語音到語音的翻譯。語音辨識技術所涉及的領域包括：訊號處理、圖型識別、概率論和資訊理論、發聲機理和聽覺機理、人工智慧等等。
</details>  
<br><br>

## Speaker-Recognition
**中文語者(聲紋)識別 (Chinese Speaker Recognition)**
> 通過聲音判別說話人身份的技術 (聲紋特徵)。
> [🌐 更多資源](https://www.twman.org/AI/ASR/SpeakerRecognition)

- **Wespeaker**
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/2210.17016v2) | [📝 v1.2.0 發布說明](https://zhuanlan.zhihu.com/p/645726183)

- **SincNet**
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/1808.00158v3)

- **實戰與教學**
  - [ASV-Subtools 聲紋識別實戰](https://speech.xmu.edu.cn/2022/1124/c18207a465302/page.htm)
  - [深度學習在聲紋識別中的應用](https://yutouwd.github.io/posts/600d0d5d/)
  - [聲紋識別原理](https://www.zhihu.com/question/30141460)
  - [CN-Celeb-AV 多模態數據集](https://zhuanlan.zhihu.com/p/647786644)
  - [提高聲紋辨識正確率 更添防疫新利器](https://www.nchc.org.tw/Message/MessageView/3731?mid=43)
  - [ICASSP 2023說話人識別方向論文合集（一）](https://zhuanlan.zhihu.com/p/645560614)
  - [相關聲紋識別介紹匯整](http://xinguiz.com/category/#/声纹识别)


<details>
<summary>2020/03/08-2020/08/29 開發心得</summary>
投入約150天。通常我們是怎樣開始項目的研究與開發？首先會先盡可能的把3年內的學術論文或比賽等SOTA都查到，然後分工閱讀找到相關的數據集和論文及相關實作；同時會找到目前已有相關產品的公司(含新創)及他們提交的專利，這部份通常再花約30天的時間；通常就是透過 Google patens、paper with codes、arxiv等等。
聲紋識別這塊在對岸查到非常多的新創公司，例如: 國音智能在我們研究開發過程就是一直被當做目標的新創公司。可以先看一下上方的DEMO影片效果；然後介紹相關實驗結果前，避免之後有人還陸續踩到我們踩過的坑；需注意的是上述很多數據集都是放在對岸像是百度雲盤等，百度是直接封鎖台灣的IP，所以你打不開是很正常的；另外像是voxcelab是切成7份，下載完再合起來也要花上不少時間，aishell、CMDS, TIMIT 比起來相對好處理就是。
簡單總結為：1. 幾種 vector 的抽取 (i-vector, d-vector, x-vector) 跟 2. 模型架構 (CNN, ResNet) 和調參，再來就是 3. 評分方式 (LDA, PLDA (Probabilistic Linear Discriminant Analysis)) 等等幾種組合；我們也使用了 kaldi 其中內附的功能，光是 kaldi 就又投入了不少時間和精力 ! 其實比起自然語言處理做聲紋識別，最小的坑莫過於雖然數據集不是很容易獲取，但是聲音是可以自行用程式加工做切割合併，然後因為場景限制，錄聲紋時的時長頗短，還得處理非註冊聲紋的處理，所以前前後後花了很多時間在將相關的數據搭配評分模式調整，也算是個大工程。

**技術指標：**
錯誤拒絕率(False Rejection Rate, FRR)：同類的兩人被系統判別為不同類。FRR為誤判案例在所有同類匹配案例中的比例
錯誤接受率(False Acceptance Rate, FAR)：不同類的兩人被系統判為同類。FAR為接受案例在所有異類匹配案例中的比例
等錯誤率(Equal Error Rate, EER)：調整threshold，當FRR=FAR時，FRR和FAR的數值稱為等錯誤率
準確率(Accuracy，ACC)：ACC=1-min(FAR+FRR)

**速度：**
Real Time Factor 實時比:衡量提取時間跟音頻時長的關係，ex:1秒可以處理80s的音頻，實時比=1:80；驗證比對速度：平均每秒能進行的聲紋比對次數
ROC曲線：描述FAR和FRR間變化的曲線，X軸為FAR,Y軸為FRR。
閥值：當分數超過閥值才做出接受決定。<br><br>  
</details>
<br><br>

## Speech-Enhancement
**中文語音增強(去噪) (Chinese Speech Enhancement)**
> 從含雜訊的語音信號中提取出純淨語音。
> [🌐 更多資源](https://www.twman.org/AI/ASR/SpeechEnhancement) | [🤗 Demo Space](https://huggingface.co/spaces/DeepLearning101/Speech-Quality-Inspection_Meta-Denoiser)

- **ClearVoice (2024-12-07)**
  - 說明：整合了語音增強與多人語者分離技術，能有效對付棘手的雞尾酒會效應（Cocktail Party Effect）。非常適合應用於多人視訊會議紀錄的聲軌拆分，或是吵雜戶外採訪音檔的極限人聲修復與提取。
  - 資源：[🐙 GitHub](https://github.com/modelscope/ClearerVoice-Studio) | [🤗 Demo](https://huggingface.co/spaces/alibabasglab/ClearVoice) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/18109659892)

- **Meta Denoiser**
  - 資源：[🐙 GitHub](https://github.com/facebookresearch/denoiser) | [📄 Real Time Speech Enhancement](https://www.alphaxiv.org/abs/2006.12847v3)

<details>
<summary>2020/08/30-2021/01/25 開發心得</summary>
分組投入約150天；說到會做語音增強(去噪音)，這一切真的只是因為那有一面之緣的圖靈獎大神在FB發文介紹FAIR的最新成果；而噪音去除你可以跟另外一個聲音分離做聯想，基本概念其實差不多，只是噪音去除是把非人聲給去除 (記得注意一下是不是多通道)；而做這個項目時，一樣也是匯整準備了相當多的學術論文和實驗結果 (如下所附) ；做語音感覺上數據也是很重要，但噪音去除相對的數據集就比較好處理，網路上都可以找到，只要進行前後調整合併，就可以產出數量頗大的數據集，唯一需要考量的就是你的 GPU 夠不夠大整個吃下了，還有你這些數據集裡的人聲是不是一樣是英文，或者是你想要中文的效果？順道一提最後我們的模型大小是經過優化的9 MB，而 RTF 是 0.08。
</details>
<br><br>

## Speaker-Separation
**中文語者分離 (Speaker Separation)**
> **定義：** 從混疊的聲音訊號中提取出單一目標訊號（解決雞尾酒會問題，即多人同時說話的場景）。
>
> 資源導航：[🌐 站長整理](https://www.twman.org/AI/ASR/SpeechSeparation) | [🤗 HF Space Demo](https://huggingface.co/spaces/DeepLearning101/Speech-Separation)

### 📚 經典論文與實作 (Papers & Code)

- **Stabilizing Label Assignment for Speech Separation**
  - 說明：Self-supervised Pre-training
  - 資源：[📄 arXiv](https://arxiv.org/abs/2010.15366) | [🐙 GitHub](https://github.com/SungFeng-Huang/SSL-pretraining-separation)

- **Sudo rm -rf**
  - 說明：Efficient Networks for Universal Audio Source Separation
  - 資源：[📄 arXiv](https://arxiv.org/abs/2007.06833) | [🐙 Code (Asteroid)](https://github.com/asteroid-team/asteroid/blob/master/asteroid/models/sudormrf.py) | [🐙 GitHub](https://github.com/etzinis/sudo_rm_rf)

- **Dual-Path Transformer (DPTNet)**
  - 說明：Direct Context-Aware Modeling for End-to-End Monaural Speech Separation
  - 資源：[📄 arXiv](https://arxiv.org/pdf/2007.13975v3.pdf)

- **Dual-path RNN (DPRNN)**
  - 說明：Efficient long sequence modeling for time-domain separation
  - 資源：[📄 arXiv](https://arxiv.org/pdf/1910.06379.pdf) | [🐙 GitHub](https://github.com/JusperLee/Dual-path-RNN-Pytorch) | [📝 閱讀筆記](https://zhuanlan.zhihu.com/p/104606356)

### 🛠️ 實戰模型與工具 (Tools)

- 2025-06-03｜**SoloSpeech**
  - 說明：一鍵提取指定說話者音訊，提升清晰度
  - 資源：[🐙 GitHub](https://github.com/WangHelin1997/SoloSpeech) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1913305854289097038)

- 2024-12-07｜**ClearVoice**
  - 說明：阿里開源一站式語音處理（降噪、分離、提取）
  - 資源：[🐙 GitHub](https://github.com/modelscope/ClearerVoice-Studio) | [🤗 Demo](https://huggingface.co/spaces/alibabasglab/ClearVoice) | [📝 中文解讀](https://juejin.cn/post/7445237715863093275)

- **其他開源項目**
  - [**TOLD**](https://zhuanlan.zhihu.com/p/650346578): 能對混疊語音建模的說話人日誌框架 (ICASSP 2023).
  - [**Mossformer**](https://zhuanlan.zhihu.com/p/609728122): 語音分離模型 (ICASSP 2023).

<details>
<summary>2020/08/30-2021/01/25 開發心得</summary>
投入約150天。如同語音踩的坑來說，比較常碰到因為網路架構在做參數調整時導致loss壞掉等等，而因數據集造成的問題少很多，網路上也比較容易找到更多的數據集，然後也有非常多的比賽有各種模型架構的結果可以參考，但是一樣是英文數據，而語音坑最好的就是只要有了像是 aishell 等的數據集，你想要切割或合併成一個語音，都不是太大的問題；例如我們就是把數據集打散混合，再從中隨機挑選兩個人，然後再從中分別挑出語音做混合；如是長度不同，選擇短者為參考，將長者切到與短者相同；最後產出約 train： 5萬多筆，約 32小時、val：1萬多筆語音，約10小時、test：9,千多筆語音，約 6小時，而這個數據集是兩兩完全重疊，後來為了處理兩兩互不完全重疊，再次另外產出了這樣的數據集：train：9萬多筆語音，計112小時、val：2萬多筆語音，計 26.3 小時、test：2萬多筆語音，計 29.4 小時。

中間也意外發現了Google brain 的 wavesplit，在有噪音及兩個人同時講話情形下，感覺效果還不差，但沒找到相關的code，未能進一步驗證或是嘗試更改數據集。還有又是那位有一起用餐之緣的深度學習大神 Yann LeCun繼發文介紹 完去噪後，又發文介紹了語音分離；後來還有像是最早應用在NLP的Transformer等Dual-path RNN (DP-RNN) 或 DPT-NET (Dual-path transformer) 等應用在語音增強/分割，另外VoiceFilter、TasNet 跟 Conv-TasNet還有sudo-rm等等也是語音分割相關，當然更不能錯過臺大電機李宏毅老師一篇SSL-pretraining-separation的論文 (務必看完臺大電機李宏毅老師的影片)，最後也是多虧李老師及第一作者黃同學的解惑，然後小夥伴們才又更深入的確認並且解決問題。
這裡做數據時相對簡單一點，直接打散混合，再從中隨機挑選兩個人，然後分別挑出語音做混合，若長度不同，選擇短者為參考，將長者切到與短者相同，兩兩完全重疊或者兩兩互不完全重疊等都對效果有不小的影響；同時也研究了Data Parallel 跟 Distributed Data Parallel 的差異，但是如何才能在 CPU 上跑得又快又準才是落地的關鍵
</details>
<br><br>

## Speech-Synthesis
**中文語音合成 (Chinese Speech Synthesis / TTS)**

### 🎙️ 2026 全球開源 TTS 語音合成與音色克隆大全 (依開發陣營分類)

*「想做有聲書、全自動短影音，還是專屬的虛擬 VTuber 聲優？目前的 TTS 技術不僅告別了傳統的『機器人平淡嗓音』，還能做到 3 秒極速複製你的聲音。本清單為你拆解目前最主流的歐美大廠方案與亞洲霸榜神作，讓你根據資安需求與硬體條件精準選型。」*

#### 1\. 歐美 AI 巨頭與國際開源社群 (資安友善 / 輕量部署篇)

如果你對專案的「原產地」有嚴格要求，或者伺服器沒有配備頂級 GPU，以下由歐美巨頭或國際社群主導的專案是你的首選：

| 模型/工具名稱 | 開發團隊/生態 | 💡 解決什麼痛點？ (核心優勢) | 🚀 推薦適用場景 & 規格標籤 |
| :--- | :--- | :--- | :--- |
| **OpenAI Edge TTS** | 🇺🇸 **微軟/OpenAI 生態** | **完全免算力、免費白嫖！** 透過呼叫微軟 Edge 瀏覽器的語音介面，免 GPU 就能產出高水準語音。 | 適合：個人開發者、輕量級網頁應用<br>`[免 GPU]` `[零成本]` `[微軟原生]` |
| **Parler-TTS** | 🇺🇸/🇪🇺 **Hugging Face** | **安裝最無腦的輕量之王**。Hugging Face 官方開源，主打「一行指令安裝」，對開發者極度友善。 | 適合：快速概念驗證 (PoC)、英/歐語系合成<br>`[極易部署]` `[國際開源]` |
| **Kokoro-TTS** | 🌐 **國際開源社群** | 近期在歐美社群討論度極高的 TTS 方案，架構輕量且聲音自然，是取代龐大模型的優質平替。 | 適合：本地端輕量化語音助理<br>`[輕量模型]` `[社群熱推]` |
| **VALL-E X** / **DragonV2.1** | 🇺🇸 **Microsoft** (微軟) | **跨語言音色保留**。微軟的經典架構（VALL-E X）與 2025 最新模型（DragonV2.1），技術底蘊深厚。 | 適合：企業級多語種配音、微軟生態系整合<br>`[大廠背書]` `[多語種克隆]` |
| **Deepgram** | 🇺🇸 **Deepgram** | **超穩定的商業級 API**。雖然不是純開源，但提供極低延遲的商業級 TTS/ASR 接口，適合不想管底層架構的企業。 | 適合：企業級 SaaS 產品、即時語音對話系統<br>`[商業方案]` `[高穩定性]` |

-----

#### 2\. 亞洲/中國開源霸榜神作 (極致擬真 & 零樣本克隆篇)

*技術客觀評析：在「中文」的表現上，以下模型目前領先全球。它們不僅能精確掌握中文的發音，甚至能生成帶有「笑聲、嘆氣、語氣詞」的超擬真人類語音。**（註：注重地緣資安的專案，建議於完全離線的本地沙盒環境中運行）***

| 模型名稱 | 開發團隊 | 💡 核心優勢與突破點 | 🚀 推薦適用場景 & 規格標籤 |
| :--- | :--- | :--- | :--- |
| **GPT-SoVITS** | 🇨🇳 **開源社群 (RVC-Boss)** | **人聲克隆的無冕王！** 只要 1 分鐘的語音樣本，就能完美複製你的聲音，目前在 GitHub 狂攬 35k+ Stars。 | 適合：VTuber 聲優克隆、個人有聲書配音<br>`[極少樣本]` `[霸榜神作]` |
| **ChatTTS** | 🇨🇳 **2noise** | **打破 AI 機械音的終極武器**。它最大的震撼在於支援在語句中加入「笑聲」、「停頓」，擬真度極高。 | 適合：AI Podcast、劇情對白生成<br>`[超高擬真]` `[情緒控制]` |
| **Fish Speech** | 🇨🇳 **Fish Audio** | **當前最火紅的全能型 TTS**。性能極其強大，不僅支援多語言克隆，還配備了視覺化的 GUI 介面，降低使用門檻。 | 適合：短影音自動化生成、多語種自媒體<br>`[多語支援]` `[自帶 GUI]` |
| **Qwen3-TTS** | 🇨🇳 **阿里巴巴** | **不只克隆，還能「捏聲音」**。提供 VoiceDesign (音色創造) 與 VoiceClone (音色克隆) 雙重強大功能。 | 適合：遊戲 NPC 配音生成、大型多模態系統<br>`[音色創造]` `[大廠開源]` |
| **Fun-CosyVoice3** | 🇨🇳 **阿里通義百聆** | **極速克隆專家**。只需短短 3 秒錄音，就能直接複製並轉換成 9 種不同的語言。 | 適合：出海行銷影片自動翻譯配音<br>`[3秒克隆]` `[跨語言]` |
| **MOSS-TTSD** / **F5-TTS** | 🇨🇳 **復旦大學 / 上海交大** | **學術界的頂規猛獸**。MOSS 經過百萬小時訓練；F5-TTS 則能用 15 秒樣本完成聲音克隆。 | 適合：學術研究、底層架構二次開發<br>`[學術開源]` `[巨量訓練]` |

-----

### 💡 額外優化建議：「效能指標」名詞解釋

> **🎯 如何評估一個 TTS 模型的好壞？**
>
>   * **Zero-Shot Cloning (零樣本克隆)：** 指模型不需要重新訓練（Fine-tuning），只需聽你講 3\~10 秒的聲音，就能直接用你的音色唸出新稿子。
>   * **Latency (延遲)：** 對於即時對話的 AI Agent，生成聲音的延遲至關重要。像 Orpheus TTS 標榜的 25ms 延遲，就是針對即時互動場景設計的。
>   * **Prosody (韻律感)：** 聲音像不像真人，關鍵在於模型能否根據上下文自動加入呼吸聲、停頓和重音，這也是 ChatTTS 等新一代模型能勝出的關鍵。

---

### ⭐ 必備明星專案 (Star Projects)

- **Fish Speech** (當前熱門)
  - 說明：性能強大，支援多語言克隆
  - 資源：[🤗 Model](https://huggingface.co/fishaudio/fish-speech-1.5) | [🐙 GitHub](https://github.com/fishaudio/fish-speech/blob/main/docs/README.zh.md) | [📖 Document](https://speech.fish.audio/zh/)
  - 延伸：[🐙 GUI版](https://github.com/AnyaCoder/fish-speech-gui) | [📝 實操教學](https://mp.weixin.qq.com/s/z8L3lpEbQ1-bkD7MM6oLsw)

- **OpenAI Edge TTS** (免費/輕量)
  - 說明：免 GPU、免費使用微軟 Edge 接口
  - 資源：[🐙 GitHub](https://github.com/travisvn/openai-edge-tts) | [📝 使用教學](https://mp.weixin.qq.com/s/lt9Vr0hR7wwyhqTh68gTkA)

- **GPT-SoVITS** (人聲克隆首選)
  - 說明：1分鐘語音訓練，35k+ Star 神級項目
  - 資源：[🐙 GitHub](https://github.com/RVC-Boss/GPT-SoVITS)

- **Deepgram** (商業方案)
  - 資源：[🌐 Official Site](https://deepgram.com/)

---

### 📅 2026 最新模型 (Latest Arrivals)

- 2026-02-22 | **Ming-flash-omni-2.0**
  - 說明：透過簡單指令即可控制產生音訊的語速、音量、音調
  - 資源：[🐙 GitHub](https://github.com/inclusionAI/Ming-omni-tts)  | [🤗 Hugging Face](https://huggingface.co/inclusionAI/Ming-flash-omni-2.0) | [📝 公眾號解讀](https://mp.weixin.qq.com/s/x3DPVL92NhO4ENm6WId-uw)

- 2026-01-26 | **Chroma 1.0**
  - 說明：專為全雙工（Full-duplex）即時語音對話設計。150ms 的超低延遲與隨時可打斷的特性，讓它成為開發次世代 AI 語音助理、虛擬陪伴機器人或即時 AI 客服智能體（Agent）的完美底層音訊引擎。
  - 資源：[🐙 GitHub](https://github.com/FlashLabs-AI-Corp/FlashLabs-Chroma)  | [🤗 Hugging Face](https://huggingface.co/FlashLabs/Chroma-4B) | [📝 公眾號解讀](https://mp.weixin.qq.com/s/V9xctkJYAuoURqbREXHidQ)

- 2025-12-24｜**Qwen3-TTS**
  - 說明：音色創造 (VoiceDesign) 與 音色克隆 (VoiceClone)
  - 資源：[📝 官方介紹](https://link.zhihu.com/?target=https%3A//www.alibabacloud.com/help/zh/model-studio/qwen-tts-voice-design%3Fspm%3Da2ty_o06.30285417.0.0.56a0c9216Ey6VM) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1987225312841445557) | [📝 公眾號解讀](https://mp.weixin.qq.com/s/4p-jiewYUkRbO6uRK-uYOQ)

- 2025-12-16｜**Fun-CosyVoice3**
  - 說明：阿里通義百聆，3秒錄音複製9種語言
  - 資源：[🐙 GitHub](https://github.com/FunAudioLLM/CosyVoice) | [📝 媒體報導](https://finance.sina.com.cn/tech/digi/2025-12-15/doc-inhawpkf1938223.shtml)

- 2025-12-12｜**VoxCPM 1.5**
  - 說明：告別機械音的「最強嘴替」
  - 資源：[🤗 HuggingFace](https://huggingface.co/openbmb/VoxCPM1.5) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1982596122645116335) | [📝 公眾號解讀](https://mp.weixin.qq.com/s/kmA6aZmCIhv1x0qS1kPCNg)

- 2025-10-12｜**NeuTTS Air**
  - 說明：主打端側運算（On-Device AI），極小的模型體積與超低功耗讓它能直接部署在 iOS/Android 手機或 IoT 穿戴裝置上。適合開發需要完全離線、保護使用者隱私的專屬語音播報 APP。
  - 資源：[🐙 GitHub](https://github.com/neuphonic/neutts-air) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1962976509611454658)

- 2025-08-15｜**ZipVoice**
  - 說明：完全擺脫對昂貴 GPU 獨立顯卡的依賴，純靠 CPU 就能實現流暢且高品質的語音合成。對於預算有限的個人開發者，或是想在輕量級雲端伺服器上部署 TTS API 服務的企業來說是絕佳的低成本方案。
  - 資源：[🐙 GitHub](https://github.com/k2-fsa/ZipVoice)

- 2025-08-08｜**KittenTTS**
  - 說明：超迷你 TTS 模型（< 25 MB）
  - 資源：[🐙 GitHub](https://github.com/KittenML/KittenTTS) | [📝 討論](https://www.reddit.com/r/LocalLLaMA/comments/1mhyzp7/kitten_tts_sota_supertiny_tts_model_less_than_25/?tl=zh-hant)

- 2025-07-30｜**Microsoft DragonV2.1**
  - 資源：[📝 官方 Blog](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/personal-voice-upgraded-to-v2-1-in-azure-ai-speech-more-expressive-than-ever-bef/4435233)

- 2025-07-25｜**Higgs Audio V2**
  - 說明：李沐團隊開源，支援越南語
  - 資源：[🐙 GitHub](https://github.com/boson-ai/higgs-audio) | [🤗 Space](https://huggingface.co/spaces/smola/higgs_audio_v2) | [🌐 Demo](https://www.boson.ai/demo/tts)
  - 延伸：[📝 李沐教學](https://zhuanlan.zhihu.com/p/1931365847840069074) | [🐙 越南語訓練](https://github.com/JimmyMa99/train-higgs-audio)

- 2025-07-23｜**FreeAudio**
  - 說明：90秒長時可控音效生成 (狼嚎、蟋蟀聲)
  - 資源：[🌐 Project](https://freeaudio.github.io/FreeAudio/) | [📝 中文解讀](https://mp.weixin.qq.com/s/gwfbwuQ91AF-WCzSVmTxNQ)

- 2025-07-14｜**MOSS-TTSD**
  - 說明：邱錫鵬團隊開源，百萬小時訓練
  - 資源：[🌐 Project](https://www.open-moss.com/en/moss-ttsd/) | [📝 媒體報導](https://finance.sina.com.cn/tech/roll/2025-07-05/doc-infemitp8423057.shtml)

- 2025-06-05｜**OpenAudio S1**
  - 說明：全球唯一高可控多語言 TTS
  - 資源：[🤗 HuggingFace](https://huggingface.co/fishaudio/openaudio-s1-mini) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1913864308212863691)

- 2025-03-30｜**MegaTTS3**
  - 說明：字節跳動開源 0.45B 參數中英雙語模型
  - 資源：[🤗 Demo](https://huggingface.co/spaces/ByteDance/MegaTTS3) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1889796359344857240)

- 2025-03-21｜**Orpheus TTS**
  - 說明：25ms 超低延遲，支援即時對話
  - 資源：[🐙 GitHub](https://github.com/canopyai/Orpheus-TTS) | [🤗 Demo](https://huggingface.co/spaces/MohamedRashad/Orpheus-TTS) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/31739692960)

- 2025-03-15｜**CSM (Conversational Speech)**
  - 說明：1B 參數實現電影級人聲
  - 資源：[🐙 GitHub](https://github.com/SesameAILabs/csm) | [📝 中文解讀](https://mp.weixin.qq.com/s/q4c1bUsRkpQHxFwpePsJLg)

- 2025-03-02｜**Spark-TTS**
  - 說明：基於單流解耦語音令牌的高效能模型
  - 資源：[🐙 GitHub](http://github.com/SparkAudio/Spark-TTS) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/29631171989)

- 2025-03-01｜**Step-Audio**
  - 說明：ComfyUI 聲音複製技術
  - 資源：[🐙 GitHub](https://github.com/stepfun-ai/Step-Audio) | [📝 中文解讀](https://mp.weixin.qq.com/s/HLYM5g8bJGCoytcoxOXzjA)

---

### 🏛️ 2024 經典模型與教程 (Classics)

- **MockingBird** (2024/11/30)
  - 說明：5秒速「復刻」聲音，35.4k Star
  - 資源：[🐙 GitHub](https://github.com/babysor/MockingBird) | [📝 中文解讀](https://mp.weixin.qq.com/s/4Ce-be5YMBTQn9aHX2OeaQ)

- **Kokoro-TTS** (2024/11/02)
  - 資源：[🤗 Demo](https://huggingface.co/spaces/hexgrad/Kokoro-TTS) | [🌐 介紹](https://kokorotts.net/zh-Hant)

- **CosyVoice & SenseVoice ComfyUI** (2024/10/15)
  - 資源：[📝 實戰教學](https://mp.weixin.qq.com/s/Lvijhi3U8jg88C8_h5Gbww)

- **F5-TTS** (2024/10/15)
  - 說明：上海交大開源，15秒克隆聲音
  - 資源：[📝 中文解讀](https://mp.weixin.qq.com/s/tWrjQfl2XkOO8GwwVyFoqw)

- **Parler-TTS** (2024/09/09)
  - 說明：Hugging Face 開源，一行指令安裝
  - 資源：[🐙 GitHub](https://github.com/huggingface/parler-tts) | [📝 中文解讀](https://mp.weixin.qq.com/s/uzYMnR6ole_RSPE_Es_thw)

- **ChatTTS** (2024/06/09)
  - 說明：支援笑聲、停頓，擬真度極高
  - 資源：[🐙 GitHub](https://github.com/2noise/ChatTTS) | [📝 部署教學](https://mp.weixin.qq.com/s/rL3vyJ_xEj7GGoKaxUh8_A)

- **VALL-E X** (2024/08/06)
  - 資源：[📝 手把手實操教學](https://mp.weixin.qq.com/s/Fo8ESzbEfjZQNUUx_giJRA)

- **MeloTTS**
  - 說明：無 GPU 也可靈活使用
  - 資源：[🐙 GitHub](https://github.com/myshell-ai/MeloTTS) | [📝 中文解讀](https://mp.weixin.qq.com/s/DSHabmduaUX5_aBedDhEFg)


## Speech-Datasets
**開源語音資料 (Speech Datasets)**

### 🇨🇳 中文與方言 / 區域性資料集

* **WenetSpeech-Chuan** (無標示日期)
* 說明：10,000小時首個大規模川渝方言語料庫，涵蓋9個領域，提供多維標註與專屬評測基準。
* 資源：[🐙 GitHub](https://github.com/ASLP-lab/WenetSpeech-Chuan)

* **Easy-Turn-Trainset** (無標示日期)
* 說明：約1100小時對話輪次檢測資料集，用於全雙工對話系統，包含完整、回應、等待等多種狀態。
* 資源：[🌐 ModelScope](https://www.modelscope.cn/datasets/ASLP-lab/Easy-Turn-Trainset)

* **WenetSpeech-Yue** (2025)
* 說明：21,800小時全球最大粵語語音資料集，涵蓋10大領域並具備多維標註，支援中英混雜場景。
* 資源：[🐙 GitHub](https://github.com/ASLP-lab/WenetSpeech-Yue)

* **Chinese-LiPS** (2025/05)
* 說明：100小時中文多模態語音辨識資料集，首創結合「唇讀資訊 + 投影片語意」，多模態融合後CER降至2.58%。
* 資源：[🌐 計畫首頁](https://kiri0824.github.io/Chinese-LiPS/)

* **CS-Dialogue** (2025/02)
* 說明：104小時目前最大的公開自發式「中英切換」對話資料集，捕捉真實自然的語言切換現象。
* 資源：[📄 arXiv](https://arxiv.org/pdf/2502.18913) | [🤗 Hugging Face](https://huggingface.co/datasets/BAAI/CS-Dialogue)

* **ChildMandarin & SeniorTalk** (2025/04)
* 說明：智源研究院發布的特殊人群資料集。ChildMandarin 填補低幼兒童（41.25小時）語音數據空白；SeniorTalk 為世界首個中文超高齡老人（55.53小時）對話資料集。
* 資源：[🌐 智源社群](https://hub.baai.ac.cn/view/44729) | [🤗 ChildMandarin](https://huggingface.co/datasets/BAAI/ChildMandarin) | [🤗 SeniorTalk](https://huggingface.co/datasets/BAAI/SeniorTalk)

* **Emilia** (2024/08)
* 說明：101,000小時目前最大多語種語音生成資料集，支援中、英等6種語言，涵蓋脫口秀、辯論等多種場景。
* 資源：[🤗 Hugging Face](https://huggingface.co/datasets/Amphion/Emilia)

* **GigaSpeech 2** (2024/06)
* 說明：30,000小時東南亞多語言（泰語、印尼語、越南語）資料集，涵蓋19個主題領域，模型效能達商業水準。
* 資源：[🤗 Hugging Face](https://huggingface.co/datasets/speechcolab/gigaspeech2)

* **LLaSO** (2024)
* 說明：開源語音大模型框架，包含1200萬對齊樣本、1350萬多任務指令樣本及標準化評估基準。
* 資源：[🐙 GitHub](https://github.com/EIT-NLP/LLaSO)

---

### 🌍 國際與特殊應用資料集

* **VietMed** (無標示日期)
* 說明：越南醫療語音資料集，包含16小時標註與2200小時無標註語音，涵蓋所有 ICD-10 疾病組及當地口音。
* 資源：[🐙 GitHub](https://github.com/leduckhai/multimed)

* **HiFiTTS-2** (2025)
* 說明：專注高頻寬（22.05kHz / 44.1kHz）語音合成的英語資料集，規模達數萬小時，支援零樣本 TTS 訓練。
* 資源：[🤗 Hugging Face](https://huggingface.co/datasets/nvidia/hifitts-2)

* **Meta Omnilingual ASR Corpus** (2025/11)
* 說明：支援1600+種語言的大規模轉錄資料集，具備少樣本學習能力（幾段音訊即可擴展），可擴展至5400+語言。
* 資源：[🐙 GitHub](https://github.com/facebookresearch/omnilingual-asr) | [🤗 Hugging Face](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus) | [🌐 線上示範](https://aidemos.atmeta.com/omnilingualasr/language-globe)

* **Common Voice** (2025/06 - 最新 v22.0)
* 說明：全球最大眾包語音資料集，累積近10萬人參與，錄製時長達3,718小時，覆蓋137種語言。
* 資源：[🌐 Mozilla Data](https://datacollective.mozillafoundation.org/datasets)

* **Bridge2AI-Voice** (2025/01)
* 說明：由 NIH 推進的醫療語音資料集，涵蓋306位參與者，針對語音障礙、神經系統疾病及憂鬱症等進行收音。
* 資源：[🏥 PhysioNet (需申請)](https://physionet.org/content/b2ai-voice/1.1/)

* **nEMO** (2024/04)
* 說明：3小時的波蘭語情緒語音資料集，包含9位演員錄製的憤怒、恐懼、快樂、悲傷等6種情緒。
* 資源：[🤗 Hugging Face](https://huggingface.co/datasets/amu-cai/nEMO) | [📄 arXiv](https://arxiv.org/abs/2404.06292)


<details 過往資訊 close>
<summary><strong>過往資訊</strong></summary>

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)：[GPT-SoVits: 上線兩天獲得了1.4k star的開源聲音克隆項目，1分鐘語音訓練TTS模型](https://zhuanlan.zhihu.com/p/679547903)

- [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

- [Rectified Flow Matching 語音合成，上海交大開源**](https://www.speechhome.com/blogs/news/1712396018944970752)：https://github.com/cantabile-kwok/VoiceFlow-TTS

- [coqui-ai TTS](https://github.com/coqui-ai/TTS)
    * [XTTS v2線上體驗](https://huggingface.co/spaces/coqui/xtts)
    * [coqui-ai TTS 簡評](https://www.speechhome.com/blogs/news/1726435660778311680)
    * [新一代開源語音庫CoQui TTS衝到了GitHub 20.5k Star](https://zhuanlan.zhihu.com/p/661291996)
- [EmotiVoice](https://github.com/netease-youdao/EmotiVoice)
    * [正式開源！網路易有道上線「易魔聲」語音合成引擎](https://zhuanlan.zhihu.com/p/666172336)

- Amphion@OpenMMLab：https://github.com/open-mmlab/Amphion
- Bark：https://github.com/suno-ai/bark
    * [最強文本轉語音工具：Bark，本地安裝+雲端部署+在線體驗詳細教程](https://zhuanlan.zhihu.com/p/630900585)
    * [使用Transformers 優化文本轉語音模型Bark](https://zhuanlan.zhihu.com/p/651951136)
    * [GitHub 開源神器Bark模型，讓文字轉語音更簡單！](https://www.speechhome.com/blogs/news/1724361984838864896)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
    * [本地訓練,開箱可用,Bert-VITS2 V2.0.2版本本地基於現有資料集訓練](https://zhuanlan.zhihu.com/p/668211415)
    * [栩栩如生,音色克隆,Bert-vits2文字轉語音打造鬼畜視訊實踐](https://zhuanlan.zhihu.com/p/662885913)
- [清華大學LightGrad-TTS，且流式實現](https://zhuanlan.zhihu.com/p/656012430)：https://github.com/thuhcsi/LightGrad

- [Wunjo AI: Synthesize & clone voices in English, Russian & Chinese](https://github.com/wladradchenko/wunjo.wladradchenko.ru)：https://huggingface.co/wladradchenko/wunjo.wladradchenko.ru

- [VALL-E：微軟全新語音合成模型可以在3秒內復制任何人的聲音](https://zhuanlan.zhihu.com/p/598473227)
    * [非官方](https://lifeiteng.github.io/valle/)：To avoid abuse, Well-trained models and services will not be provided.
- [BLSTM-RNN、Deep Voice、Tacotron…你都掌握了吗？一文总结语音合成必备经典模型（一）](https://new.qq.com/rain/a/20221204A02GIT00)

- [Tacotron2、GST、Glow-TTS、Flow-TTS…你都掌握了吗？一文总结语音合成必备经典模型（二）](https://cloud.tencent.com/developer/article/2250062)

- [出門問問MeetVoice, 讓合成聲音以假亂真](https://zhuanlan.zhihu.com/p/92903377)
</details>
