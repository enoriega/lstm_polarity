import dynet as dy
from utils import Instance

if __name__ == "__main__":

    p = "model.dy"
    pc = dy.ParameterCollection()

    # a = pc.load_param(p, "/vanilla-lstm-builder/_0")
    # b = pc.load_param(p, "/vanilla-lstm-builder/_1")
    # c = pc.load_param(p, "/vanilla-lstm-builder/_2")

    # print(a.value().sum())
    # print(b.value().sum())
    # print(sum(c.value()))
    builder = dy.LSTMBuilder(1, 100, 10, pc)

    x = pc.parameters_list()

    y = 1

    sentence = "To formally prove that increased ROS levels enhance anti-tumour effects of the SG-free diet , the authors crossed Emu-Myc mice with mice deficient for Tigar , a fructose-2 ,6-bisphosphatase , which limits glycolysis and favours pentose phosphate pathways , thus limiting ROS levels XREF_BIBR , XREF_BIBR ( XREF_FIG ) .".lower()

    words = Instance.normalize(sentence)
    a = 1

    #x = builder.initial_state().b()
    #a = 1
