# Synthesize(inputs, outputs):
# plist := set of all terminals
# while(true):
# plist := grow(plist);
# plist := elimEquvalents(plist, inputs);
# forall( p in plist)
# if(isCorrect(p, inputs, outputs)): return p;


# TODO: bottom-up enumerative synthesis
def synthesize(inputs: list[str], outputs: list[str]):
    pass
