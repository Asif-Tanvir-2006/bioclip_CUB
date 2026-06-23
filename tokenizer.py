import json

token2eng = {}
eng2token = {}
with open('tokens.json', 'r') as f:
    lih = json.load(f)

print(len(lih))
for i in range(len(lih)):
    token2eng[i] = lih[i]
    eng2token[lih[i]] =  i
def tokenize(text):
    l = 0
    r = 0
    tokens = []
    while(r<len(text)):
        curr = ""
        while(not(curr in eng2token)):
            curr+=text[r]
            r+=1
            # print(curr)
        tokens.append(eng2token[curr])
        l = r
    return tokens

def detokenize(toks):
    ans = []
    for tok in toks:
        ans.append(token2eng[tok])
    return "".join(ans)
