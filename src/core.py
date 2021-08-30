from src.geographical_clustering import GeographicalClustering
from src.textual_clustering import TextualClustering
from src.visual_clustering import VisualClustering


def main():
    # gc = GeographicalClustering()
    # gc.process()
    # vc = VisualClustering(1)
    # vc.process()
    tc = TextualClustering(1, 0)
    tc.process()


if __name__ == '__main__':
    main()
