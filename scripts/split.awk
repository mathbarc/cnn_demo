BEGIN {srand()}
{f = FILENAME (rand() <= 0.7 ? "_train" : "_test"); print > f}