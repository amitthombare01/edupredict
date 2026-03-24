"""Entry point for Student Performance Analysis."""

from src import analysis


def main() -> None:
    """Run the main analysis flow."""
    print("Loading data...")
    df = analysis.load_student_data("../data/student_performance.csv")
    print(f"Loaded {len(df)} rows")


if __name__ == "__main__":
    main()
