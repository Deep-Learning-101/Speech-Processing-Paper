# Speech Processing (語音處理)

[那些語音處理 (Speech Processing) 踩的坑](https://blog.twman.org/2021/04/ASR.html)

語音辨識（speech recognition）技術，也被稱為自動語音辨識（英語：Automatic Speech Recognition, ASR）、電腦語音識別（英語：Computer Speech Recognition）或是語音轉文字識別（英語：Speech To Text, STT），其目標是以電腦自動將人類的語音內容轉換為相應的文字；跟小夥伴們一起嘗試過NEMO還有Kaldi、MASR、VOSK，wav2vec以及Google、Azure等API，更別說後來陸續又出現SpeechBrain、出門問問的WeNet跟騰訊PIKA等。目前已知可訓練聲學模型(AM)中文語音(中國發音/用語，可惜還沒臺灣較靠譜的)公開數據如：Magic-Data_Mandarin-Chinese-Read-Speech-Corpus、aidatatang、aishell-1 、aishell-2等約2000多小時(aishell目前已到4，但想商用至少得破萬小時較靠譜)；再搭配語言模型(LM)，然後基於各種演算法架構優化各有優缺點，效果也各有優劣。與說話人辨識及說話人確認不同，後者嘗試辨識或確認發出語音的說話人而非其中所包含的詞彙內容。 語音辨識技術的應用包括語音撥號、語音導航、室內裝置控制、語音文件檢索、簡單的聽寫資料錄入等。語音辨識技術與其他自然語言處理技術如機器翻譯及語音合成技術相結合，可以構建出更加複雜的應用，例如語音到語音的翻譯。語音辨識技術所涉及的領域包括：訊號處理、圖型識別、概率論和資訊理論、發聲機理和聽覺機理、人工智慧等等。

https://www.twman.org/AI/ASR

https://huggingface.co/DeepLearning101

#
# 中文語者(聲紋)識別 (Chinese Speaker Recognition)

https://www.twman.org/AI/ASR/SpeakerRecognition

找到描述特定對象的聲紋特徵，通過聲音判別說話人身份的技術；借助不同人的聲音，在語譜圖的分佈情況不同這一特徵，去對比兩個人的聲音，來判斷是否同人。

### **技術指標**
#### 錯誤拒絕率(False Rejection Rate, FRR)：同類的兩人被系統判別為不同類。FRR為誤判案例在所有同類匹配案例中的比例
#### 錯誤接受率(False Acceptance Rate, FAR)：不同類的兩人被系統判為同類。FAR為接受案例在所有異類匹配案例中的比例
#### 等錯誤率(Equal Error Rate, EER)：調整threshold，當FRR=FAR時，FRR和FAR的數值稱為等錯誤率
#### 準確率(Accuracy，ACC)：ACC=1-min(FAR+FRR)
### **速度：**
#### Real Time Factor 實時比:衡量提取時間跟音頻時長的關係，ex:1秒可以處理80s的音頻，實時比=1:80；驗證比對速度：平均每秒能進行的聲紋比對次數
#### ROC曲線：描述FAR和FRR間變化的曲線，X軸為FAR,Y軸為FRR。
#### 閥值：當分數超過閥值才做出接受決定。<br><br>  


(2020/03/08-2020/08/29) 投入約150天。通常我們是怎樣開始項目的研究與開發？首先會先盡可能的把3年內的學術論文或比賽等SOTA都查到，然後分工閱讀找到相關的數據集和論文及相關實作；同時會找到目前已有相關產品的公司(含新創)及他們提交的專利，這部份通常再花約30天的時間；通常就是透過 Google patens、paper with codes、arxiv等等。
聲紋識別這塊在對岸查到非常多的新創公司，例如: 國音智能在我們研究開發過程就是一直被當做目標的新創公司。可以先看一下上方的DEMO影片效果；然後介紹相關實驗結果前，避免之後有人還陸續踩到我們踩過的坑；需注意的是上述很多數據集都是放在對岸像是百度雲盤等，百度是直接封鎖台灣的IP，所以你打不開是很正常的；另外像是voxcelab是切成7份，下載完再合起來也要花上不少時間，aishell、CMDS, TIMIT 比起來相對好處理就是。
簡單總結為：1. 幾種 vector 的抽取 (i-vector, d-vector, x-vector) 跟 2. 模型架構 (CNN, ResNet) 和調參，再來就是 3. 評分方式 (LDA, PLDA (Probabilistic Linear Discriminant Analysis)) 等等幾種組合；我們也使用了 kaldi 其中內附的功能，光是 kaldi 就又投入了不少時間和精力 ! 其實比起自然語言處理做聲紋識別，最小的坑莫過於雖然數據集不是很容易獲取，但是聲音是可以自行用程式加工做切割合併，然後因為場景限制，錄聲紋時的時長頗短，還得處理非註冊聲紋的處理，所以前前後後花了很多時間在將相關的數據搭配評分模式調整，也算是個大工程。

### [提高聲紋辨識正確率 更添防疫新利器](https://www.nchc.org.tw/Message/MessageView/3731?mid=43)

### [ASV-Subtools聲紋識別實戰](https://speech.xmu.edu.cn/2022/1124/c18207a465302/page.htm)

### [聲紋識別原理](https://www.zhihu.com/question/30141460)

### [深度學習在聲紋識別中的應用](https://yutouwd.github.io/posts/600d0d5d/)

### [相關聲紋識別介紹匯整](http://xinguiz.com/category/#/声纹识别)

## SincNet：Speaker Recognition from Raw Waveform with SincNet

https://arxiv.org/abs/1808.00158

https://github.com/mravanelli/SincNet


#
# 中文語音增強(去噪) Chinese Speech Enhancement

https://www.twman.org/AI/ASR/SpeechEnhancement

找到描述特定聲音特徵，並將其去除以提高質量；從含雜訊的語音信號中提取出純淨語音的過程

(2020/08/30-2021/01/25) 分組投入約150天；說到會做語音增強(去噪音)，這一切真的只是因為那有一面之緣的圖靈獎大神在FB發文介紹FAIR的最新成果；而噪音去除你可以跟另外一個聲音分離做聯想，基本概念其實差不多，只是噪音去除是把非人聲給去除 (記得注意一下是不是多通道)；而做這個項目時，一樣也是匯整準備了相當多的學術論文和實驗結果 (如下所附) ；做語音感覺上數據也是很重要，但噪音去除相對的數據集就比較好處理，網路上都可以找到，只要進行前後調整合併，就可以產出數量頗大的數據集，唯一需要考量的就是你的 GPU 夠不夠大整個吃下了，還有你這些數據集裡的人聲是不是一樣是英文，或者是你想要中文的效果？順道一提最後我們的模型大小是經過優化的9 MB，而 RTF 是 0.08。

## Real Time Speech Enhancement in the Waveform Domain
https://arxiv.org/pdf/2006.12847.pdf

https://github.com/facebookresearch/denoiser

https://www.youtube.com/watch?v=77cm_MVtLfk

#
# 中文語者分離(分割) Chinese Speech Separation (Speaker Separation)

https://www.twman.org/AI/ASR/SpeechSeparation

從多個聲音信號中提取出目標信號；多個說話人情況的語音辨識問題，比如雞尾酒會上很多人講話

(2020/08/30-2021/01/25) 投入約150天。如同語音踩的坑來說，比較常碰到因為網路架構在做參數調整時導致loss壞掉等等，而因數據集造成的問題少很多，網路上也比較容易找到更多的數據集，然後也有非常多的比賽有各種模型架構的結果可以參考，但是一樣是英文數據，而語音坑最好的就是只要有了像是 aishell 等的數據集，你想要切割或合併成一個語音，都不是太大的問題；例如我們就是把數據集打散混合，再從中隨機挑選兩個人，然後再從中分別挑出語音做混合；如是長度不同，選擇短者為參考，將長者切到與短者相同；最後產出約 train： 5萬多筆，約 32小時、val：1萬多筆語音，約10小時、test：9,千多筆語音，約 6小時，而這個數據集是兩兩完全重疊，後來為了處理兩兩互不完全重疊，再次另外產出了這樣的數據集：train：9萬多筆語音，計112小時、val：2萬多筆語音，計 26.3 小時、test：2萬多筆語音，計 29.4 小時。

中間也意外發現了Google brain 的 wavesplit，在有噪音及兩個人同時講話情形下，感覺效果還不差，但沒找到相關的code，未能進一步驗證或是嘗試更改數據集。還有又是那位有一起用餐之緣的深度學習大神 Yann LeCun繼發文介紹 完去噪後，又發文介紹了語音分離；後來還有像是最早應用在NLP的Transformer等Dual-path RNN (DP-RNN) 或 DPT-NET (Dual-path transformer) 等應用在語音增強/分割，另外VoiceFilter、TasNet 跟 Conv-TasNet還有sudo-rm等等也是語音分割相關，當然更不能錯過臺大電機李宏毅老師一篇SSL-pretraining-separation的論文 (務必看完臺大電機李宏毅老師的影片)，最後也是多虧李老師及第一作者黃同學的解惑，然後小夥伴們才又更深入的確認並且解決問題。
這裡做數據時相對簡單一點，直接打散混合，再從中隨機挑選兩個人，然後分別挑出語音做混合，若長度不同，選擇短者為參考，將長者切到與短者相同，兩兩完全重疊或者兩兩互不完全重疊等都對效果有不小的影響；同時也研究了Data Parallel 跟 Distributed Data Parallel 的差異，但是如何才能在 CPU 上跑得又快又準才是落地的關鍵

## Sudo rm -rf: Efficient Networks for Universal Audio Source Separation

https://github.com/asteroid-team/asteroid/blob/master/asteroid/models/sudormrf.py

https://github.com/etzinis/sudo_rm_rf