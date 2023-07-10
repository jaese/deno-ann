# Generate text with a character-based RNN

This example is to test RNN layers in `neural/recurrent/`.

It is based on [Text generation with an RNN](https://www.tensorflow.org/text/tutorials/text_generation) guide on the Tensorflow website. Given a sequence of characters of training data, the script trains a simple RNN model to predict the next character in the sequence.

You can use the following commands to download the training data ("Shakespeare").

```sh
mkdir -p _local
curl -L https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt -o _local/shakespeare.txt

head -n 1000 _local/shakespeare.txt > _local/shakespeare_small_train.txt
tail -n 500 _local/shakespeare.txt > _local/shakespeare_small_test.txt
```

## How to run

```sh
deno run train.ts
```

After one epoch of training, the model generates texts more or less like this:

```
Predict on "Citizen":  thas yound bous sour and gour sherene the sou the hend itheser tore noethe,
Thall therore thatheess soun wore ther ous thand the ares lou an thath thamind ar thancou ghe thou thethen tous sour this hours fomer andound mand ouren soble bold sous on thourime therou the owe the than it the an ford anden courer nou four hou than the hou thour mather our Cout,
Thous ores our ant ourile our gouss and in thathe therile the theow ang thond torin thers than thind mous theereres me thitt mor the thars in the thours sour thou theril onde youn themen mouss than hawsours wethe thar mind ther tit sans thalled ond then then the thingers thales.

ARIUS:
Mou wout ans thetherthet thard thind pous hand
Morche dou cound thoun fas and me and and onthe hour ing, hour thise thetrirs orin the has than hathe the oule sende chetheer that out ar of toust the thours mou orebe thit ours an in harp thar the I themar thir therer yous wanthe.
```
