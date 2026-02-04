# AprilTag.msg
int32 id              # Tag ID
float32[2] center     # Centre [x, y] en pixels
float32[8] corners    # 4 coins en pixels [x0,y0,x1,y1,x2,y2,x3,y3]
float32 hamming       # Hamming distance (confiance)