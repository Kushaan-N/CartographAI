function appendLine(text, type = 'default') {
  const colors = {
    action: '#58a6ff', success: '#3fb950', error: '#da3633',
    hint: '#d29922', coverage: '#bc8cff', default: '#8b949e'
  };
  const div = document.createElement('div');
  div.style.color = colors[type] || colors.default;
  div.style.marginBottom = '2px';
  div.textContent = `[${new Date().toISOString().substr(11,8)}] ${text}`;
  document.getElementById('terminal-output').appendChild(div);
  autoScroll();
}

function autoScroll() {
  const panel = document.getElementById('terminal-panel');
  panel.scrollTop = panel.scrollHeight;
}
