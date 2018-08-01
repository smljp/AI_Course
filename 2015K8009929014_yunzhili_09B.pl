subset_self([],[]).
subset_self([X|STail],[X|Tail]):- subset_self(STail,Tail).
subset_self(STail,[_Head|Tail]):- subset_self(STail,Tail).
