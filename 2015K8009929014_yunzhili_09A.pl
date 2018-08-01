member_self(X,[_H|T]):- member_self(X,T).
member_self(X,[X|_T]).


