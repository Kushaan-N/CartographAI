const STATUS_COLORS = {
  hypothesized: '#8b949e', probing: '#d29922',
  discovered: '#238636', failed: '#da3633', mapped: '#1f6feb'
};

let simulation, svgEl, g, linkGroup, nodeGroup;
let nodes = [], links = [];

function initSimulation() {
  svgEl = document.getElementById('graph-svg');
  const svg = d3.select(svgEl);
  svg.append('defs').append('marker')
    .attr('id', 'arrowhead').attr('viewBox', '0 -5 10 10')
    .attr('refX', 20).attr('refY', 0).attr('markerWidth', 6).attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', '#30363d');

  g = svg.append('g');
  linkGroup = g.append('g').attr('class', 'links');
  nodeGroup = g.append('g').attr('class', 'nodes');

  svg.call(d3.zoom().on('zoom', e => g.attr('transform', e.transform)));

  simulation = d3.forceSimulation()
    .force('link', d3.forceLink().id(d => d.id).distance(100))
    .force('charge', d3.forceManyBody().strength(-250))
    .force('center', d3.forceCenter(svgEl.clientWidth / 2, svgEl.clientHeight / 2))
    .on('tick', ticked);
}

function ticked() {
  linkGroup.selectAll('line')
    .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
    .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
  nodeGroup.selectAll('g.node')
    .attr('transform', d => `translate(${d.x},${d.y})`);
}

function renderGraph() {
  const link = linkGroup.selectAll('line').data(links, d => `${d.source.id||d.source}-${d.target.id||d.target}`);
  link.enter().append('line')
    .attr('stroke', '#30363d').attr('stroke-width', 1.5)
    .attr('stroke-dasharray', d => d.confirmed ? 'none' : '4 2')
    .attr('marker-end', 'url(#arrowhead)')
    .merge(link);
  link.exit().remove();

  const node = nodeGroup.selectAll('g.node').data(nodes, d => d.id);
  const nodeEnter = node.enter().append('g').attr('class', 'node')
    .call(d3.drag()
      .on('start', (event, d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
      .on('drag', (event, d) => { d.fx=event.x; d.fy=event.y; })
      .on('end', (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }));
  nodeEnter.append('circle').attr('r', 14);
  nodeEnter.append('text').attr('dy', 24).attr('text-anchor', 'middle')
    .attr('fill', '#8b949e').attr('font-size', '11px');

  nodeGroup.selectAll('g.node').select('circle')
    .attr('fill', d => STATUS_COLORS[d.status] || '#8b949e')
    .attr('stroke', '#0d1117').attr('stroke-width', 2);
  nodeGroup.selectAll('g.node').select('text').text(d => d.id);
  node.exit().remove();

  simulation.nodes(nodes);
  simulation.force('link').links(links);
  simulation.alpha(0.3).restart();
}

function updateGraph(data) {
  if (data.nodes) nodes = data.nodes;
  if (data.edges) links = data.edges.map(e => ({...e, source: e.source, target: e.target}));
  renderGraph();

  if (data.coverage !== undefined) {
    document.getElementById('coverage-fill').style.width = (data.coverage * 100) + '%';
    document.getElementById('coverage-label').textContent = `Coverage: ${(data.coverage * 100).toFixed(1)}%`;
  }
  if (data.level !== undefined) {
    document.getElementById('level-label').textContent = `Level ${data.level}`;
  }
}

function connectWebSocket() {
  const ws = new WebSocket(`ws://${window.location.host}/ws`);
  ws.onmessage = function(event) {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'graph' || data.nodes) {
        updateGraph(data);
      }
      if (data.type === 'terminal') {
        appendLine(data.text, data.line_type || 'default');
      }
      if (data.type === 'coverage') {
        const pct = (data.coverage * 100).toFixed(1);
        document.getElementById('coverage-fill').style.width = pct + '%';
        document.getElementById('coverage-label').textContent = 'Coverage: ' + pct + '%';
        if (data.level) {
          document.getElementById('level-label').textContent = 'Level ' + data.level;
        }
      }
    } catch(err) { console.error('WS parse error', err); }
  };
  ws.onclose = () => setTimeout(connectWebSocket, 2000);
}

document.addEventListener('DOMContentLoaded', () => { initSimulation(); connectWebSocket(); });
