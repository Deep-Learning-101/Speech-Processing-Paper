# Speech Processing (語音處理)

https://www.twman.org/AI/ASR

https://huggingface.co/DeepLearning101

[那些語音處理 (Speech Processing) 踩的坑](https://blog.twman.org/2021/04/ASR.html)

[音視頻開發基礎入門｜聲音的採集與量化、音頻數字信號質量、音頻碼率](https://zhuanlan.zhihu.com/p/577850804)

[語音識別資料匯總：常見庫和特徵對比](https://zhuanlan.zhihu.com/p/616020595)

[Mozilla Common Voice Datasets - zhTW](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/zh-TW)

#
# 中文語音識別 (Chinese Speech Recognition)

### RelatedLink

* **Whisper**
    * [Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf)
    * [Introducing Whisper](https://openai.com/research/whisper)
    * [Whisper-Finetune](https://github.com/yeyupiaoling/Whisper-Finetune)
    * [免費的即時語音轉文字工具：Whisper Live，精準高效，支援多語言](https://www.zhihu.com/tardis/zm/art/676939649)：https://github.com/collabora/WhisperLive
    * [語音辨識的未來已來-探索Distil-Whisper，輕量級AI的強大力量](https://zhuanlan.zhihu.com/p/666238999)
    * [Insanely Fast Whisper：超快速的Whisper語音辨識腳本](https://www.wehelpwin.com/article/4532)：https://github.com/Vaibhavs10/insanely-fast-whisper
    * [微調Whisper語音辨識模型與加速推理](https://github.com/yeyupiaoling/Whisper-Finetune/)
    * [Whisper: openAI開源準確率最高的通用語言語音識別](https://zhuanlan.zhihu.com/p/634462613)
    * [使用Transformers 為多語種語音識別任務微調Whisper 模型](https://huggingface.co/blog/zh/fine-tune-whisper)
    * [在消費級顯示卡上微調OpenAI開源的自動語言辨識模型Whisper：8GB記憶體即可針對自己的資料建立ASR模型](https://www.datalearner.com/blog/1051684336082480)
    * [WhisperX](https://github.com/m-bain/whisperX)
    * [Faster-Whisper對影片進行雙語字幕轉錄實踐(Python3.10)](https://zhuanlan.zhihu.com/p/664892334)
* **FunASR**
    * [FunASR: A Fundamental End-to-End Speech Recognition Toolkit](https://arxiv.org/abs/2305.11013)
    * https://github.com/alibaba-damo-academy/FunASR
    * [阿里達摩院開源大型端到端語音識別工具包FunASR](https://zhuanlan.zhihu.com/p/634646731)
    * [達摩院FunASR離線文件轉寫SDK發布](https://zhuanlan.zhihu.com/p/642807244)
* **WeNet**：[58同城：WeNet端到端語音識別大規模落地方案](https://zhuanlan.zhihu.com/p/573133117)
    * [WeNet: Production First and Production Ready End-to-End Speech Recognition Toolkit](https://arxiv.org/pdf/2102.01547.pdf)
* [**PaddleSpeech**](https://github.com/PaddlePaddle/PaddleSpeech)
* [**Speech Brain**：A PyTorch-based Speech Toolkit](https://github.com/speechbrain/speechbrain)
* [**Kaldi 2**：FSA/FST algorithms, differentiable, with PyTorch compatibility.](https://github.com/k2-fsa/k2)
    * [Next-gen-Kaldi 近期進展](https://zhuanlan.zhihu.com/p/617877445)
* [QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions](https://arxiv.org/pdf/1910.10261.pdf)
* [Self-training and Pre-training are Complementary for Speech Recognition](https://arxiv.org/pdf/2010.11430.pdf)
* [Meta Massively Multilingual Speech, MMS](https://github.com/facebookresearch/fairseq)：https://huggingface.co/facebook/mms-tts-eng


### **2020/03-2021/01 開發心得：**
語音辨識（speech recognition）技術，也被稱為自動語音辨識（英語：Automatic Speech Recognition, ASR）、電腦語音識別（英語：Computer Speech Recognition）或是語音轉文字識別（英語：Speech To Text, STT），其目標是以電腦自動將人類的語音內容轉換為相應的文字；跟小夥伴們一起嘗試過NEMO還有Kaldi、MASR、VOSK，wav2vec以及Google、Azure等API，更別說後來陸續又出現SpeechBrain、出門問問的WeNet跟騰訊PIKA等。目前已知可訓練聲學模型(AM)中文語音(中國發音/用語，可惜還沒臺灣較靠譜的)公開數據如：Magic-Data_Mandarin-Chinese-Read-Speech-Corpus、aidatatang、aishell-1 、aishell-2等約2000多小時(aishell目前已到4，但想商用至少得破萬小時較靠譜)；再搭配語言模型(LM)，然後基於各種演算法架構優化各有優缺點，效果也各有優劣。與說話人辨識及說話人確認不同，後者嘗試辨識或確認發出語音的說話人而非其中所包含的詞彙內容。 語音辨識技術的應用包括語音撥號、語音導航、室內裝置控制、語音文件檢索、簡單的聽寫資料錄入等。語音辨識技術與其他自然語言處理技術如機器翻譯及語音合成技術相結合，可以構建出更加複雜的應用，例如語音到語音的翻譯。語音辨識技術所涉及的領域包括：訊號處理、圖型識別、概率論和資訊理論、發聲機理和聽覺機理、人工智慧等等。


#
# 中文語者(聲紋)識別 (Chinese Speaker Recognition)

https://www.twman.org/AI/ASR/SpeakerRecognition

找到描述特定對象的聲紋特徵，通過聲音判別說話人身份的技術；借助不同人的聲音，在語譜圖的分佈情況不同這一特徵，去對比兩個人的聲音，來判斷是否同人。

### **相關論文**
* [Wespeaker: A Research and Production oriented Speaker Embedding Learning Toolkit](https://arxiv.org/pdf/2210.17016.pdf)
* [SincNet：Speaker Recognition from Raw Waveform with SincNet](https://arxiv.org/abs/1808.00158)

### **相關連結**
* [Wespeaker v1.2.0 發布：新增SSL Recipe，NIST SRE 數據集支持, PLDA 及自適應代碼等](https://zhuanlan.zhihu.com/p/645726183)
* [ASV-Subtools聲紋識別實戰](https://speech.xmu.edu.cn/2022/1124/c18207a465302/page.htm)
* [ICASSP 2023說話人識別方向論文合集（一）](https://zhuanlan.zhihu.com/p/645560614)
* [聲紋識別原理](https://www.zhihu.com/question/30141460)
* [深度學習在聲紋識別中的應用](https://yutouwd.github.io/posts/600d0d5d/)
* [相關聲紋識別介紹匯整](http://xinguiz.com/category/#/声纹识别)
* [提高聲紋辨識正確率 更添防疫新利器](https://www.nchc.org.tw/Message/MessageView/3731?mid=43)
* [CN-Celeb-AV: 多場景視聽多模態數據集發布](https://zhuanlan.zhihu.com/p/647786644)

### **2020/03/08-2020/08/29 開發心得：**
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

#
# 中文語音增強(去噪) Chinese Speech Enhancement

https://www.twman.org/AI/ASR/SpeechEnhancement

找到描述特定聲音特徵，並將其去除以提高質量；從含雜訊的語音信號中提取出純淨語音的過程

### **相關論文**
* [Real Time Speech Enhancement in the Waveform Domain](https://arxiv.org/pdf/2006.12847.pdf)

### **相關連結**
* https://github.com/facebookresearch/denoiser

* https://www.youtube.com/watch?v=77cm_MVtLfk

### **2020/08/30-2021/01/25 開發心得：**
分組投入約150天；說到會做語音增強(去噪音)，這一切真的只是因為那有一面之緣的圖靈獎大神在FB發文介紹FAIR的最新成果；而噪音去除你可以跟另外一個聲音分離做聯想，基本概念其實差不多，只是噪音去除是把非人聲給去除 (記得注意一下是不是多通道)；而做這個項目時，一樣也是匯整準備了相當多的學術論文和實驗結果 (如下所附) ；做語音感覺上數據也是很重要，但噪音去除相對的數據集就比較好處理，網路上都可以找到，只要進行前後調整合併，就可以產出數量頗大的數據集，唯一需要考量的就是你的 GPU 夠不夠大整個吃下了，還有你這些數據集裡的人聲是不是一樣是英文，或者是你想要中文的效果？順道一提最後我們的模型大小是經過優化的9 MB，而 RTF 是 0.08。


#
# 中文語者分離(分割) Chinese Speech Separation (Speaker Separation)

https://www.twman.org/AI/ASR/SpeechSeparation

從多個聲音信號中提取出目標信號；多個說話人情況的語音辨識問題，比如雞尾酒會上很多人講話

### **相關論文**

* Stabilizing Label Assignment for Speech Separation by Self-supervised Pre-training：https://arxiv.org/abs/2010.15366
    * https://github.com/SungFeng-Huang/SSL-pretraining-separation
* Self-supervised Pre-training Reduces Label Permutation Instability of Speech Separation：https://arxiv.org/pdf/2010.15366v1.pdf
    * https://github.com/SungFeng-Huang/SSL-pretraining-separation
* Sudo rm -rf: Efficient Networks for Universal Audio Source Separation：https://arxiv.org/abs/2007.06833
    * https://github.com/asteroid-team/asteroid/blob/master/asteroid/models/sudormrf.py 
    * https://github.com/etzinis/sudo_rm_rf   
* Dual-Path Transformer Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation：https://arxiv.org/pdf/2007.13975v3.pdf
* Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation：https://arxiv.org/pdf/1910.06379.pdf
    * https://github.com/JusperLee/Dual-path-RNN-Pytorch
    * [閱讀筆記”Dual-path RNN for Speech Separation“](https://zhuanlan.zhihu.com/p/104606356)

### **相關連結**

* [ICASSP2023論文代碼開源｜TOLD能對混疊語音建模的說話人日誌框架](https://zhuanlan.zhihu.com/p/650346578)
* [ICASSP 2023論文模型開源｜語音分離Mossformer](https://zhuanlan.zhihu.com/p/609728122)

### **2020/08/30-2021/01/25 開發心得：**

投入約150天。如同語音踩的坑來說，比較常碰到因為網路架構在做參數調整時導致loss壞掉等等，而因數據集造成的問題少很多，網路上也比較容易找到更多的數據集，然後也有非常多的比賽有各種模型架構的結果可以參考，但是一樣是英文數據，而語音坑最好的就是只要有了像是 aishell 等的數據集，你想要切割或合併成一個語音，都不是太大的問題；例如我們就是把數據集打散混合，再從中隨機挑選兩個人，然後再從中分別挑出語音做混合；如是長度不同，選擇短者為參考，將長者切到與短者相同；最後產出約 train： 5萬多筆，約 32小時、val：1萬多筆語音，約10小時、test：9,千多筆語音，約 6小時，而這個數據集是兩兩完全重疊，後來為了處理兩兩互不完全重疊，再次另外產出了這樣的數據集：train：9萬多筆語音，計112小時、val：2萬多筆語音，計 26.3 小時、test：2萬多筆語音，計 29.4 小時。

中間也意外發現了Google brain 的 wavesplit，在有噪音及兩個人同時講話情形下，感覺效果還不差，但沒找到相關的code，未能進一步驗證或是嘗試更改數據集。還有又是那位有一起用餐之緣的深度學習大神 Yann LeCun繼發文介紹 完去噪後，又發文介紹了語音分離；後來還有像是最早應用在NLP的Transformer等Dual-path RNN (DP-RNN) 或 DPT-NET (Dual-path transformer) 等應用在語音增強/分割，另外VoiceFilter、TasNet 跟 Conv-TasNet還有sudo-rm等等也是語音分割相關，當然更不能錯過臺大電機李宏毅老師一篇SSL-pretraining-separation的論文 (務必看完臺大電機李宏毅老師的影片)，最後也是多虧李老師及第一作者黃同學的解惑，然後小夥伴們才又更深入的確認並且解決問題。
這裡做數據時相對簡單一點，直接打散混合，再從中隨機挑選兩個人，然後分別挑出語音做混合，若長度不同，選擇短者為參考，將長者切到與短者相同，兩兩完全重疊或者兩兩互不完全重疊等都對效果有不小的影響；同時也研究了Data Parallel 跟 Distributed Data Parallel 的差異，但是如何才能在 CPU 上跑得又快又準才是落地的關鍵

#
# 中文語音合成 Chinese Speech Synthesis

### **相關連結**

* [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)：[GPT-SoVits: 上線兩天獲得了1.4k star的開源聲音克隆項目，1分鐘語音訓練TTS模型](https://zhuanlan.zhihu.com/p/679547903)
* [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
* [Rectified Flow Matching 語音合成，上海交大開源**](https://www.speechhome.com/blogs/news/1712396018944970752)：https://github.com/cantabile-kwok/VoiceFlow-TTS
* [coqui-ai TTS](https://github.com/coqui-ai/TTS)
    * [XTTS v2線上體驗](https://huggingface.co/spaces/coqui/xtts)
    * [coqui-ai TTS 簡評](https://www.speechhome.com/blogs/news/1726435660778311680)
    * [新一代開源語音庫CoQui TTS衝到了GitHub 20.5k Star](https://zhuanlan.zhihu.com/p/661291996)
* [EmotiVoice](https://github.com/netease-youdao/EmotiVoice)
    * [正式開源！網路易有道上線「易魔聲」語音合成引擎](https://zhuanlan.zhihu.com/p/666172336)
* Amphion@OpenMMLab：https://github.com/open-mmlab/Amphion
* **Bark**：https://github.com/suno-ai/bark
    * [最強文本轉語音工具：Bark，本地安裝+雲端部署+在線體驗詳細教程](https://zhuanlan.zhihu.com/p/630900585)
    * [使用Transformers 優化文本轉語音模型Bark](https://zhuanlan.zhihu.com/p/651951136)
    * [GitHub 開源神器Bark模型，讓文字轉語音更簡單！](https://www.speechhome.com/blogs/news/1724361984838864896)
* [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
    * [本地訓練,開箱可用,Bert-VITS2 V2.0.2版本本地基於現有資料集訓練](https://zhuanlan.zhihu.com/p/668211415)
    * [栩栩如生,音色克隆,Bert-vits2文字轉語音打造鬼畜視訊實踐](https://zhuanlan.zhihu.com/p/662885913)
* [**清華大學LightGrad-TTS，且流式實現**](https://zhuanlan.zhihu.com/p/656012430)：https://github.com/thuhcsi/LightGrad
* [**Wunjo AI: Synthesize & clone voices in English, Russian & Chinese**](https://github.com/wladradchenko/wunjo.wladradchenko.ru)：https://huggingface.co/wladradchenko/wunjo.wladradchenko.ru
* [VALL-E：微軟全新語音合成模型可以在3秒內復制任何人的聲音](https://zhuanlan.zhihu.com/p/598473227)
    * [非官方](https://lifeiteng.github.io/valle/)：To avoid abuse, Well-trained models and services will not be provided.


* [BLSTM-RNN、Deep Voice、Tacotron…你都掌握了吗？一文总结语音合成必备经典模型（一）](https://new.qq.com/rain/a/20221204A02GIT00)
* [Tacotron2、GST、Glow-TTS、Flow-TTS…你都掌握了吗？一文总结语音合成必备经典模型（二）](https://cloud.tencent.com/developer/article/2250062)

* [出門問問MeetVoice, 讓合成聲音以假亂真](https://zhuanlan.zhihu.com/p/92903377)
