"""Funzioni di supporto per i notebook didattici sulle porte logiche
(neurone lineare → neurone affine → rete con attivazione)."""


def identity(x):
    """Funzione identità: f(x) = x. Restituisce l'input invariato.

    Usata come "attivazione neutra" per dimostrare didatticamente che una
    rete profonda senza non-linearità è matematicamente equivalente a un
    singolo neurone affine: la composizione di mappe affini è ancora una
    mappa affine.

    Quando lo studente sostituisce `attivazione = identity` con
    `attivazione = jnp.tanh` (o un'altra non-linearità), la stessa rete
    riesce finalmente a imparare problemi non linearmente separabili
    come XOR.
    """
    return x
