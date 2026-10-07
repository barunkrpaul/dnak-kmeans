"""Patch the Ball k-means C++ release: double precision and a CLI main()."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
for a, b in [("typedef float OurType;", "typedef double OurType;"),
             ("typedef VectorXf VectorOur;", "typedef VectorXd VectorOur;"),
             ("typedef MatrixXf MatrixOur;", "typedef MatrixXd MatrixOur;")]:
    assert a in s, a
    s = s.replace(a, b)
i = s.index("int main(int argc, char* argv[]) {")
s = s[:i] + '''int main(int argc, char* argv[]) {
    // usage: ballkm data.csv centroids.csv isRing labels_out
    MatrixOur dataset = load_data(argv[1]);
    MatrixOur centroids = load_data(argv[2]);
    bool ring = atoi(argv[3]) != 0;
    VectorXi labels = ring ? ball_k_means_Ring(dataset, centroids, true)
                           : ball_k_means_noRing(dataset, centroids, true);
    std::ofstream out(argv[4]);
    for (int i = 0; i < labels.size(); i++) out << labels(i) << "\\n";
    return 0;
}
'''
open(dst, "w").write(s)
