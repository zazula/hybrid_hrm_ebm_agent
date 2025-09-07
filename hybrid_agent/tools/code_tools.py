
import re, subprocess, tempfile, os, json

def syntax_ok(patch_text, language='python'):
    if language == 'python':
        try:
            compile(patch_text, '<string>', 'exec')
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"
    # extend for other languages
    return True, None

def run_pytests(repo_path, timeout=120):
    try:
        out = subprocess.run(['pytest','-q'], cwd=repo_path, capture_output=True, timeout=timeout, text=True)
        return out.returncode == 0, out.stdout + '\n' + out.stderr
    except Exception as e:
        return False, str(e)

def patch_applies(repo_path, file_relpath, new_content):
    abs_path = os.path.join(repo_path, file_relpath)
    try:
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True, None
    except Exception as e:
        return False, str(e)
