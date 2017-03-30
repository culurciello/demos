# Experiments to get better results


## Only center crop (commit: 2afed36):

eugenioculurciello@Pora:~/FWDNXT/github/demos/video-localization(master) $ python3 demo.py -c /Users/eugenioculurciello/Desktop/Marsh/video2-small.srt -i /Users/eugenioculurciello/Desktop/Marsh/video2-2-small.mp4 
Demo captioning for Video Localization
This video has 511 frames
Frames per second: 30.0
Loaded (458, 512) embeddings
Error: 9.58904109589041 %


## Center, right, left crop (commit: 31a7be1):

eugenioculurciello@Pora:~/FWDNXT/github/demos/video-localization(master) $ python3 demo.py -c /Users/eugenioculurciello/Desktop/Marsh/video2-small.srt -i /Users/eugenioculurciello/Desktop/Marsh/video2-2-small.mp4 
Demo captioning for Video Localization
This video has 511 frames
Frames per second: 30.0
Loaded (458, 1536) embeddings
Error: 1.36986301369863 %