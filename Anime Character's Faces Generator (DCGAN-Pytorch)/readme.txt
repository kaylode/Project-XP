This is a Generative Adversarial Networks model which is used for generate samples from Anime Character Faces Dataset found online.
The program was written with Python, used Pytorch framework

Datasets: Anime Faces - 60000 picture samples - dimensions varied
Framework: Pytorch

Samples were gathered by a written image crawler, which automatically download faces from websites, then applied anime face detection to crop out faces for training data.
The data was resized to size 64x64 then feed to the DCGANs.

ML Algorithm applied:
	Model: DCGANs (added one-sided label smoothing, weight initiation)
	Optimizer: Adam with betas
	Loss: Binary Cross-Entropy
	Other: GPU
