from collections import OrderedDict

import torch
import torch.nn as nn
import math


class UNet(nn.Module):

	def __init__(self, in_channels=3, out_channels=1, init_features=32):
		super(UNet, self).__init__()

		features = init_features
		# 128 x 128 x in_channels to 128 x 128 x features
		self.encoder1 = UNet._block(in_channels, features, name="enc1")
		# 128 x 128 x features to 64 x 64 x features
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		# 64 x 64 x features to 64 x 64 x features*2
		self.encoder2 = UNet._block(features, features * 2, name="enc2")
		# 64 x 64 x features*2 to 32 x 32 x features*4
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
		# 32 x 32 x features*4 to 16 x 16 x features*8
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
		# 16 x 16 x features*8 to 8 x 8 x features*8
		self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

		# 8 x 8 x features*8 to 8 x 8 x features*16
		self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

		# 8 x 8 x features*16 to 16 x 16 x features*8
		self.upconv4 = nn.ConvTranspose2d(
			features * 16, features * 8, kernel_size=2, stride=2
		)
		self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
		# 16 x 16 x features*8 to 32 x 32 x features*4
		self.upconv3 = nn.ConvTranspose2d(
			features * 8, features * 4, kernel_size=2, stride=2
		)
		self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
		# 32 x 32 x features*4 to 64 x 64 x features*2
		self.upconv2 = nn.ConvTranspose2d(
			features * 4, features * 2, kernel_size=2, stride=2
		)
		self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
		# 64 x 64 x features*2 to 128 x 128 x features
		self.upconv1 = nn.ConvTranspose2d(
			features * 2, features, kernel_size=2, stride=2
		)
		self.decoder1 = UNet._block(features * 2, features, name="dec1")

		# 128 x 128 x features to 128 x 128 x out_channels
		self.conv = nn.Conv2d(
			in_channels=features, out_channels=out_channels, kernel_size=1
		)

	def forward(self, x):
		enc1 = self.encoder1(x)
		enc2 = self.encoder2(self.pool1(enc1))
		enc3 = self.encoder3(self.pool2(enc2))
		enc4 = self.encoder4(self.pool3(enc3))

		bottleneck = self.bottleneck(self.pool4(enc4))

		dec4 = self.upconv4(bottleneck)
		dec4 = torch.cat((dec4, enc4), dim=1)
		dec4 = self.decoder4(dec4)
		dec3 = self.upconv3(dec4)
		dec3 = torch.cat((dec3, enc3), dim=1)
		dec3 = self.decoder3(dec3)
		dec2 = self.upconv2(dec3)
		dec2 = torch.cat((dec2, enc2), dim=1)
		dec2 = self.decoder2(dec2)
		dec1 = self.upconv1(dec2)
		dec1 = torch.cat((dec1, enc1), dim=1)
		dec1 = self.decoder1(dec1)
		#return self.conv(dec1)
		return torch.sigmoid(self.conv(dec1))  # use this from the old unet because output in 0~1

	@staticmethod
	def _block(in_channels, features, name):
		return nn.Sequential(
			OrderedDict(
				[
					(
						name + "conv1",
						nn.Conv2d(
							in_channels=in_channels,
							out_channels=features,
							kernel_size=3,
							padding=1,
							bias=False,
						),
					),
					(name + "norm1", nn.BatchNorm2d(num_features=features)),
					(name + "relu1", nn.ReLU(inplace=True)),
					(
						name + "conv2",
						nn.Conv2d(
							in_channels=features,
							out_channels=features,
							kernel_size=3,
							padding=1,
							bias=False,
						),
					),
					(name + "norm2", nn.BatchNorm2d(num_features=features)),
					(name + "relu2", nn.ReLU(inplace=True)),
				]
			)
		)

def get_positional_encoding(max_seq_len, embedding_features, embsize1, embsize2):
	d_model = embedding_features * embsize1 * embsize2
	pe = torch.zeros(max_seq_len, d_model)
	for pos in range(max_seq_len):
		for i in range(0, d_model, 2):
			pe[pos, i] = \
				math.sin(pos / (10000 ** ((2 * i) / d_model)))
			pe[pos, i + 1] = \
				math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
	return pe.reshape(max_seq_len, embedding_features, embsize1, embsize2)


class UNetWithEmbeddingGen(nn.Module):

	def __init__(self, in_channels=3, out_channels=1, init_features=32, \
				 embedding_features=1, video_len=1000, embsize=8, fix_emb=True):
		super(UNetWithEmbeddingGen, self).__init__()

		features = init_features
		# 128 x 128 x in_channels to 128 x 128 x features
		self.encoder1 = UNet._block(in_channels, features, name="enc1")
		# 128 x 128 x features to 64 x 64 x features
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		# 64 x 64 x features to 64 x 64 x features*2
		self.encoder2 = UNet._block(features, features * 2, name="enc2")
		# 64 x 64 x features*2 to 32 x 32 x features*4
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
		# 32 x 32 x features*4 to 16 x 16 x features*8
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
		# 16 x 16 x features*8 to 8 x 8 x features*8
		self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

		# 8 x 8 x (features*8 + embedding_features) to 8 x 8 x features*16
		self.bottleneck = UNet._block(features * 8 + embedding_features, \
									  features * 16, name="bottleneck")

		# 8 x 8 x features*16 to 16 x 16 x features*8
		self.upconv4 = nn.ConvTranspose2d(
			features * 16, features * 8, kernel_size=2, stride=2
		)
		self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
		# 16 x 16 x features*8 to 32 x 32 x features*4
		self.upconv3 = nn.ConvTranspose2d(
			features * 8, features * 4, kernel_size=2, stride=2
		)
		self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
		# 32 x 32 x features*4 to 64 x 64 x features*2
		self.upconv2 = nn.ConvTranspose2d(
			features * 4, features * 2, kernel_size=2, stride=2
		)
		self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
		# 64 x 64 x features*2 to 128 x 128 x features
		self.upconv1 = nn.ConvTranspose2d(
			features * 2, features, kernel_size=2, stride=2
		)
		self.decoder1 = UNet._block(features * 2, features, name="dec1")

		# 128 x 128 x features to 128 x 128 x out_channels
		self.conv = nn.Conv2d(
			in_channels=features, out_channels=out_channels, kernel_size=1
		)

		self.embsize = embsize
		if fix_emb:
			self.frame_embed = nn.Parameter(get_positional_encoding(video_len, embedding_features, embsize, 8),\
										requires_grad=False)
		else:
			self.frame_embed = nn.Parameter(torch.randn(video_len, embedding_features, embsize, 8),\
										requires_grad=True)
		self.embedding_features = embedding_features

	def forward(self, x, indices):
		ind0 = indices.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
		ind1 = ind0.repeat(1, self.embedding_features, self.embsize, 8)

		# print('self.frame_embed.size()=', self.frame_embed.size())
		# print('ind1.size()=', ind1.size())
		# print('indices.size()=', indices.size())

		embed = torch.gather(self.frame_embed, 0, ind1)

		enc1 = self.encoder1(x)
		enc2 = self.encoder2(self.pool1(enc1))
		enc3 = self.encoder3(self.pool2(enc2))
		enc4 = self.encoder4(self.pool3(enc3))

		cat_enc4 = torch.cat((self.pool4(enc4), embed), dim=1)

		bottleneck = self.bottleneck(cat_enc4)

		dec4 = self.upconv4(bottleneck)
		dec4 = torch.cat((dec4, enc4), dim=1)
		dec4 = self.decoder4(dec4)
		dec3 = self.upconv3(dec4)
		dec3 = torch.cat((dec3, enc3), dim=1)
		dec3 = self.decoder3(dec3)
		dec2 = self.upconv2(dec3)
		dec2 = torch.cat((dec2, enc2), dim=1)
		dec2 = self.decoder2(dec2)
		dec1 = self.upconv1(dec2)
		dec1 = torch.cat((dec1, enc1), dim=1)
		dec1 = self.decoder1(dec1)
		#return self.conv(dec1)
		return torch.sigmoid(self.conv(dec1))  # use this from the old unet because output in 0~1

	@staticmethod
	def _block(in_channels, features, name):
		return nn.Sequential(
			OrderedDict(
				[
					(
						name + "conv1",
						nn.Conv2d(
							in_channels=in_channels,
							out_channels=features,
							kernel_size=3,
							padding=1,
							bias=False,
						),
					),
					(name + "norm1", nn.BatchNorm2d(num_features=features)),
					(name + "relu1", nn.ReLU(inplace=True)),
					(
						name + "conv2",
						nn.Conv2d(
							in_channels=features,
							out_channels=features,
							kernel_size=3,
							padding=1,
							bias=False,
						),
					),
					(name + "norm2", nn.BatchNorm2d(num_features=features)),
					(name + "relu2", nn.ReLU(inplace=True)),
				]
			)
		)

