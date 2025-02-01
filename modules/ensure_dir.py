import os


def ensure_dir(dir_path: str, ensure_exist: bool = True) -> None:
    """
    指定されたディレクトリが存在しない場合、ensure_exist が True なら作成し、
    False なら例外を発生させる。

    Args:
        dir_path (str): チェックするディレクトリのパス。
        ensure_exist (bool): 存在しない場合に作成するかどうか（デフォルトは True）。

    Raises:
        FileNotFoundError: ensure_exist が False で、ディレクトリが存在しない場合に発生。
    """
    if not os.path.exists(dir_path):
        if ensure_exist:
            os.makedirs(dir_path, exist_ok=True)
        else:
            raise FileNotFoundError(f"Directory does not exist: {dir_path}")
