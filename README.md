![My image](https://github.com/IraKorshunova/folk-rnn/blob/master/folkrnn_logo.png)

# How to Use:

**data_loader.py:** 
Uses train_src, train_tgt, valid_src and valid_tgt data files. The train_src and valid_src files contain the original data missing titles. The train_tgt and valid_tgt files contain the left shifted data with an <eos> token at the end of each tune. 

# Folk music style modelling using LSTMs

This code was used for the following published works:

1. Sturm and Ben-Tal, ["Taking the Models back to Music Practice: Evaluating Generative Transcription Models built using Deep Learning,”](http://jcms.org.uk/issues/Vol2Issue1/taking-models-back-to-music-practice/article.html) J. Creative Music Systems, Vol. 2, No. 1, Sep. 2017.

1. Sturm, Santos, Ben-Tal and Korshunova, "Music transcription modelling and composition using deep learning", in Proc. [1st Conf. Computer Simulation of Musical Creativity](https://csmc2016.wordpress.com), Huddersfield, UK, July 2016.

1. Sturm, Santos and Korshunova, ["Folk Music Style Modelling by Recurrent Neural Networks with Long Short Term Memory Units"](http://ismir2015.uma.es/LBD/LBD13.pdf), Late-breaking demo at the 2015 Int. Symposium on Music Information Retrieval

4. The folk-rnn v1, v2 and v3 Session Books https://highnoongmt.wordpress.com/2018/01/05/volumes-1-20-of-folk-rnn-v1-transcriptions/

11. 47,000+ tunes at The Endless folk-rnn Traditional Music Session http://www.eecs.qmul.ac.uk/~sturm/research/RNNIrishTrad/index.html 

# Music compositions resulting from versions of this code:

1. "Bastard Tunes" by Oded Ben-Tal + folk-rnn (v2) (2017) https://www.youtube.com/playlist?list=PLdTpPwVfxuXpQ03F398HH463SAE0vR2X8
1. "Safe Houses" by Úna Monaghan + folk-rnn (v2) (for concertina and tape, 2017) https://youtu.be/x6LS9MbQj7Y
1. "Dialogues with folk-rnn" by Luca Tuchet + folk-rnn (v2) (for smart mandolin, 2017) https://youtu.be/pkf3VqPieoo
1. "The Fortootuise Pollo" by Bob L. Sturm + folk-rnn (v1) (2017) https://soundcloud.com/sturmen-1/the-fortootuise-pollo-1
3. "March to the Mainframe" by Bob L. Sturm + folk-rnn (v2) (2017) https://youtu.be/TLzBcMvl15M?list=PLdTpPwVfxuXrdOyjtwfokrpzfpIlnJc5o Score is here: https://highnoongmt.files.wordpress.com/2017/12/twoshortpieceswithaninterlude.pdf
4. "Interlude" by Bob L. Sturm + folk-rnn (v2) (2017) https://youtu.be/NZ08dDdYh3U?list=PLdTpPwVfxuXrdOyjtwfokrpzfpIlnJc5o (synthesized version: https://soundcloud.com/sturmen-1/interlude-synthesised) Score is here: https://highnoongmt.files.wordpress.com/2017/12/twoshortpieceswithaninterlude.pdf
5. "The Humours of Time Pigeon" by Bob L. Sturm + folk-rnn (v1) (2017) https://youtu.be/1xBisQK8-3E?list=PLdTpPwVfxuXrdOyjtwfokrpzfpIlnJc5o (synthesized version: https://soundcloud.com/sturmen-1/the-humours-time-pigeon-synthesised) Score is here: https://highnoongmt.files.wordpress.com/2017/12/twoshortpieceswithaninterlude.pdf
6. "Carol of the Cells" by Bob L. Sturm + folk-rnn (v2) (2017) https://highnoongmt.wordpress.com/2017/12/16/carol-of-the-cells-from-the-ai-to-the-orchestra/
1. "Chicken Bits and Bits and Bobs" by Bob L. Sturm + folk-rnn (v1) (2017) https://youtu.be/n-avS-ozrqU Score is here: https://highnoongmt.files.wordpress.com/2017/04/sturm_chicken.pdf
6. "The Ranston Cassock" by Bob L. Sturm + folk-rnn (v1) (2016) https://youtu.be/JZ-47IavYAU (Version for viola and tape: https://highnoongmt.wordpress.com/2017/06/18/the-ranston-cassock-take-2/)
2. Tunes by folk-rnn harmonised by DeepBach (2017)
    1. "The Glas Herry Comment" by folk-rnn (v1) + DeepBach (2017) https://youtu.be/y9xJl-ljOuA
    1. "The Drunken Pint" by folk-rnn (v1) + DeepBach (2017) https://youtu.be/xJyp7vBNVA0
    1. X:633 by folk-rnn (v2) + DeepBach (2017) https://youtu.be/BUIrbZS5eXc
    1. X:7153 by folk-rnn (v2) + DeepBach (2017) https://youtu.be/tdKCzAyynu4

5. Tunes from [folk-rnn v1 session volumes](https://highnoongmt.wordpress.com/2018/01/05/volumes-1-20-of-folk-rnn-v1-transcriptions/)
    1. "The Cunning Storm" adapted and performed by Bob L. Sturm and Carla Sturm (2017) https://highnoongmt.wordpress.com/2018/01/07/the-cunning-storm-a-folk-rnn-v1-original/
    1. "The Irish Show" adapted and performed by Bob L. Sturm (2017) https://highnoongmt.wordpress.com/2017/11/23/folk-rnn-v1-tune-the-irish-show/
    1. "Sean No Cottifall" adapted and performed by Bob L. Sturm (2017) https://highnoongmt.wordpress.com/2017/11/11/folk-rnn-v1-tune-sean-no-cottifall/
    1. "Optoly Louden" adapted and performed by Bob L. Sturm (2017) https://highnoongmt.wordpress.com/2017/07/01/optoly-louden-a-folk-rnn-original/
    1. "Bonny An Ade Nullway" adapted and performed by Bob L. Sturm (2017) https://highnoongmt.wordpress.com/2017/09/18/bobby-an-ade-nullway-a-folk-rnn-v1-tune/
    1. "The Drunken Pint" adapted and performed by Bob L. Sturm (2017) https://youtu.be/omHhyVD3PD8; performed by EECSers (2017) https://youtu.be/0gosLln8Org
    1. "The Glas Herry Comment" adapted and performed by Bob L. Sturm (2017) https://youtu.be/QZh0WSjFFDs; performed by EECSers (2017) https://youtu.be/NiUAZBLh2t0
    1. "The Mal's Copporim" adapted and performed by Bob L. Sturm (2016) https://youtu.be/YMbWwU2JdLg; performed by EECSers (2017) https://youtu.be/HOPz71Bx714
    1. "The Castle Star" adapted and performed by Bob L. Sturm (2015) https://highnoongmt.wordpress.com/2015/08/12/deep-learning-for-assisting-the-process-of-music-composition-part-2/
    1. "The Doutlace" adapted and performed by Bob L. Sturm (2015) https://highnoongmt.wordpress.com/2015/08/11/deep-learning-for-assisting-the-process-of-music-composition-part-1/

5. Tunes from the [folk-rnn v2 session volumes](https://highnoongmt.wordpress.com/2018/01/05/volumes-1-20-of-folk-rnn-v1-transcriptions/)
    1. Transcriptions 1469, 1470 & 1472 performed by Torbjorn Hultmark (2016) https://youtu.be/4kLxvJ-rXDs
    1. X:488 performed by Bob L. Sturm (2017) https://youtu.be/QWvlnOqlSes; performed by EECSers (2017) https://youtu.be/QWvlnOqlSes
    2. X:4542 adapted and performed by Bob L. Sturm (2017) https://highnoongmt.wordpress.com/2017/10/06/folk-rnn-v2-tune-4542/
    3. X:2857 adapted and performed by Bob L. Sturm and Carla Sturm (2017) https://highnoongmt.wordpress.com/2017/12/02/folk-rnn-v2-tune-2857/

5. Tunes from the [folk-rnn v3 session volumes](https://highnoongmt.wordpress.com/2018/01/05/volumes-1-20-of-folk-rnn-v1-transcriptions/)
    1. "The 2714 Polka" adapted and performed by Bob L. Sturm (2017) https://highnoongmt.wordpress.com/2017/09/10/the-2714-polka-a-folk-rnn-original/
    1. X:1166 adapted and performed by Bob L. Sturm (2017) https://highnoongmt.wordpress.com/2017/09/16/folk-rnn-v3-tune-1166/
    1. X:1650 adapted and performed by Bob L. Sturm (2017) https://highnoongmt.wordpress.com/2017/09/17/folk-rnn-v3-tune-1650/
    1. X:6197 adapted and performed by Bob L. Sturm (2017) https://highnoongmt.wordpress.com/2017/09/25/folk-rnn-v3-tune-6197/
    2. X:8589 (A Derp Deep Learning Ditty) adapted and performed by Bob L. Sturm (2017) https://highnoongmt.wordpress.com/2017/10/08/a-derp-deep-learning-ditty/

6. "A Windy Canal" by folk-rnn (v2) and Bob L. Sturm (2017) https://soundcloud.com/sturmen-1/a-windy-canal
7. "Experimental lobotomy of a deep network with subsequent stimulation (2)" by folk-rnn (v2 with lobotomy) and Bob L. Sturm, Carla Sturm (2017) https://soundcloud.com/sturmen-1/experimental-lobotomy-of-a-deep-network-with-subsequent-stimulation-2
6. "It came out from a pretrained net" by Bob L. Sturm + folk-rnn (v2) (2016) https://highnoongmt.wordpress.com/2016/12/24/taking-a-christmas-carol-toward-the-dodecaphonic-by-derp-learning/
6. "The Millennial Whoop Reel" by Bob L. Sturm + folk-rnn (v2) (2016) https://highnoongmt.wordpress.com/2016/08/29/millennial-whoop-with-derp-learning-a-reel/
6. "The Millennial Whoop Jig" by Bob L. Sturm + folk-rnn (v2) (2016) https://highnoongmt.wordpress.com/2016/08/28/millennial-whoop-with-derp-learning/
6. "Eight Short Outputs ..." by folk-rnn (v1) + Bob L. Sturm (2015) https://highnoongmt.wordpress.com/2015/12/20/eight-short-outputs-now-on-youtube/
7. “We three layers o’ hidd’n units are” by Bob L. Sturm + folk-rnn (v2) (2015) https://highnoongmt.wordpress.com/2015/12/16/tis-the-season-for-some-deep-carols/
8. "The March of Deep Learning" by Bob L. Sturm + folk-rnn (v1) (2015) https://highnoongmt.wordpress.com/2015/08/15/deep-learning-for-assisting-the-process-of-music-composition-part-4/

# Media
1. March 31 2017, ‘Machine folk’ music composed by AI shows technology’s creative side, The Conversation (https://theconversation.com/machine-folk-music-composed-by-ai-shows-technologys-creative-side-74708)
2. Dec. 23 2017 "AI Has Been Creating Music and the Results Are...Weird" PC Mag (http://uk.pcmag.com/news/92577/ai-has-been-creating-music-and-the-results-areweird)
2. Nov. 18, 2017 Le Tube avec Stéphane Bern et Laurence Bloch, France http://www.canalplus.fr/emissions/pid8584-le-tube.html
2. June 3 2017, "An A.I. in London is Writing Its Own Music and It Sounds Heavenly" https://www.inverse.com/article/32276-folk-music-ai-folk-rnn-musician-s-best-friend
2. June 8 2017, "Computer program created to write Irish trad tunes" http://www.irishtimes.com/business/technology/computer-program-created-to-write-irish-trad-tunes-1.3112238
3. June 19 2017 "Folk-RNN is the Loquantur Rhythm artist of June" (providing music for phone call waits) https://zc1.campaign-view.com/ua/SharedView?od=11287eca6b3187&cno=11a2b0b20c9c037&cd=12a539b2f47976f3&m=4 (Here is a sample: https://highnoongmt.wordpress.com/2017/06/17/deep-learning-on-hold/)
3. June 18 2017, "Real Musicians Evaluate Music Made by Artificial Intelligence" https://motherboard.vice.com/en_us/article/irish-folk-music-ai
4. June 1 2017, "Can an AI Machine Hold Copyright Protection Over Its Work?" https://artlawjournal.com/ai-machine-copyright/
2. May 26 2017 The Daily Mail named our project "Bot Dylan" (http://www.dailymail.co.uk/sciencetech/article-4544400/Researchers-create-computer-writes-folk-music.html), and then didn't even link to this page. Plus the video they edited has no computer-generated music in it. Well done!
2. April 13 2017 “Eine Maschine meistert traditionelle Folk-Music” http://www.srf.ch/kultur/netzwelt/eine-maschine-meistert-traditionelle-folk-music 
