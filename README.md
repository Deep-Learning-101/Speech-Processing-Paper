#
https://www.twman.org/AI/ASR

https://huggingface.co/DeepLearning101

https://deep-learning-101.github.io/Speech-Processing
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

## Speech-Processing
**語音處理 (Speech Processing)**

- 2025-09-23：[Sherpa onnx](https://github.com/k2-fsa/sherpa-onnx)
- 2025-05-14：[ten-turn-detection](https://zread.ai/TEN-framework/ten-turn-detection)；[ten-vad](https://zread.ai/TEN-framework/ten-vad)
- 2025-01-19：[小米語音首席科學家Daniel Povey：語音辨識捲完了，下一個機會在哪裡？](https://www.jiqizhixin.com/articles/2025-01-19-4?)
-  [ASR/TTS 開發避坑指南：語音辨識與合成的常見挑戰與對策](https://blog.twman.org/2024/02/asr-tts.html)；[探討 ASR 和 TTS 技術應用中的問題，強調數據質量的重要性](https://deep-learning-101.github.io/asr-tts)
-  [那些語音處理踩的坑](https://blog.twman.org/2021/04/ASR.html)；[分享語音處理領域的實務經驗，強調資料品質對模型效果的影響](https://deep-learning-101.github.io/speech)
- [音視頻開發基礎入門｜聲音的採集與量化、音頻數字信號質量、音頻碼率](https://zhuanlan.zhihu.com/p/577850804)
- [一文總覽萬字語音合成系列基礎與論文總結](https://mp.weixin.qq.com/s/S9T9fk9THUF3JQRnNuOM7Q)
- [Mozilla Common Voice Datasets - zhTW](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/zh-TW)
- [語音識別資料匯總：常見庫和特徵對比](https://zhuanlan.zhihu.com/p/616020595)
- [語音合成,語音辨識常見資料集](https://mp.weixin.qq.com/s/xGAEzuT5x7BkTRH6DCJFhA)
- [2024年-2025年開源語音資料彙整：數十萬小時多語種、兒童老人語音、醫療健康等（截止2025年11月）](https://zhuanlan.zhihu.com/p/1974579913194501708)


**中文語音識別 (Chinese Speech Recognition)**
> 通過語音信號處理和模式識別讓機器自動識別和理解人類的口述。
> [🌐 更多 ASR 資源](https://www.twman.org/AI/ASR)

### 🔥 最新模型 (2025)

- 2025-12-23｜**MedASR**
  - 說明：Google 發布醫學語音辨識模型
  - 資源：[🤗 HuggingFace](https://huggingface.co/google/medasr)

- 2025-12-16｜**Fun-ASR**
  - 說明：阿里開源 0.8B 模型，效能直逼 12B 巨頭
  - 資源：[🐙 GitHub](https://github.com/FunAudioLLM/Fun-ASR) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1984310683358217029)

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

---

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
  - 說明：一站式語音處理工具包 (降噪、分離、提取)
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

### 📅 2025 最新模型 (Latest Arrivals)

- 2025-12-24｜**Qwen3-TTS**
  - 說明：音色創造 (VoiceDesign) 與 音色克隆 (VoiceClone)
  - 資源：[📝 官方介紹](https://link.zhihu.com/?target=https%3A//www.alibabacloud.com/help/zh/model-studio/qwen-tts-voice-design%3Fspm%3Da2ty_o06.30285417.0.0.56a0c9216Ey6VM) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1987225312841445557)

- 2025-12-16｜**Fun-CosyVoice3**
  - 說明：阿里通義百聆，3秒錄音複製9種語言
  - 資源：[🐙 GitHub](https://github.com/FunAudioLLM/CosyVoice) | [📝 媒體報導](https://finance.sina.com.cn/tech/digi/2025-12-15/doc-inhawpkf1938223.shtml)

- 2025-12-12｜**VoxCPM 1.5**
  - 說明：告別機械音的「最強嘴替」
  - 資源：[🤗 HuggingFace](https://huggingface.co/openbmb/VoxCPM1.5) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1982596122645116335)

- 2025-10-12｜**NeuTTS Air**
  - 說明：手機也能跑，3秒克隆聲音
  - 資源：[🐙 GitHub](https://github.com/neuphonic/neutts-air) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1962976509611454658)

- 2025-08-15｜**ZipVoice**
  - 說明：CPU is all you need!
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
