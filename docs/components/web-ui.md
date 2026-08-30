# Web UI Module

Browser-based dashboard for monitoring and control.

## Overview

Located in `web/` directory. Provides:
- Real-time training dashboard
- Model management interface
- Dataset browser
- System monitoring
- User-friendly controls

## Components

### index.html — Main Page

Complete web interface with sections:
- Navigation header
- Training dashboard
- Model gallery
- Dataset browser
- Settings panel
- Logs viewer

### globals.css — Global Styles

Application-wide styling:
- Color scheme and theming
- Typography and fonts
- Layout utilities
- Responsive design
- Dark/light mode support

### theme.css — Theme Configuration

Customizable theme variables:
- Primary/secondary colors
- Font families
- Spacing scales
- Border radius
- Shadow effects

See `THEME_MIGRATION.md` for theme system details.

## Features

### Dashboard

Real-time monitoring of training:
- Live loss curve
- Accuracy graph
- GPU/CPU usage
- Memory tracking
- Training speed (samples/sec)
- ETA countdown

### Model Management

Browse and manage models:
- Import new models
- List downloaded models
- View model details
- Download model info
- Delete unused models
- Export trained models

### Dataset Browser

Inspect and manage datasets:
- View dataset statistics
- Check class distribution
- Validate data quality
- Preview images
- Edit labels
- Create new datasets

### Job Monitoring

Track training jobs:
- List active jobs
- View job history
- Cancel running job
- Resume paused job
- Download logs
- Export results

### Settings

Configure application:
- API connection settings
- Model download paths
- Default hyperparameters
- Notification preferences
- Dark/light mode
- Language selection

## Architecture

### Frontend Stack

```
├── HTML5
│   └── Semantic structure
├── CSS3
│   ├── Flexbox/Grid
│   ├── Animations
│   └── Responsive design
└── Vanilla JavaScript
    ├── API calls
    ├── Real-time updates
    └── State management
```

### Backend Integration

Connects to API server on `localhost:8000`:

```javascript
// Fetch training jobs
fetch('http://localhost:8000/train/list')
  .then(r => r.json())
  .then(jobs => updateDashboard(jobs))

// Start new training
fetch('http://localhost:8000/train/start', {
  method: 'POST',
  body: JSON.stringify({
    model: 'resnet50',
    dataset: 'my-data',
    epochs: 10
  })
})
```

### Real-Time Updates

WebSocket connection for live metrics:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/metrics');

ws.onmessage = (event) => {
  const metrics = JSON.parse(event.data);
  updateMetrics(metrics);
};
```

## Usage

### Running Locally

1. Start API server:
```bash
cd serving
python main.py
```

2. Open web browser:
```
http://localhost:8000
```

OR serve static files separately:
```bash
python -m http.server 8080 --directory web
# Open http://localhost:8080
```

### Docker Deployment

Web is served by API server:
```bash
docker-compose up
# Web available at http://localhost:8000
```

## Customization

### Changing Theme

Edit `theme.css`:

```css
:root {
  --color-primary: #007bff;      /* Change primary color */
  --color-secondary: #6c757d;    /* Change secondary color */
  --font-family: 'Roboto', sans-serif;  /* Change font */
  --spacing-unit: 8px;           /* Change spacing scale */
}
```

### Adding New Dashboard Widget

1. Add HTML:
```html
<div class="widget" id="my-widget">
  <h3>My Widget</h3>
  <div class="content">Loading...</div>
</div>
```

2. Add CSS:
```css
#my-widget {
  grid-column: span 2;
  padding: var(--spacing-lg);
}
```

3. Add JavaScript:
```javascript
function updateMyWidget(data) {
  document.getElementById('my-widget').innerHTML = data;
}

// Call periodically
setInterval(() => {
  fetch('/api/my-data').then(r => r.json()).then(updateMyWidget);
}, 1000);
```

## Performance

- **Load time:** <2 seconds
- **Update frequency:** 1 second (real-time)
- **Supported browsers:** Chrome, Firefox, Safari, Edge
- **Mobile support:** Responsive on tablets
- **Offline support:** Some features work offline (cached data)

## Accessibility

- Semantic HTML5
- ARIA labels
- Keyboard navigation
- Color contrast WCAG AA
- Focus indicators
- Screen reader support

## Security

- CSRF protection
- JWT token validation
- Input sanitization
- XSS prevention
- Secure headers (CSP, X-Frame-Options)
- HTTPS only in production

## Troubleshooting

**Blank page:**
- Check browser console for errors
- Verify API server is running
- Check CORS settings

**Slow dashboard:**
- Reduce update frequency
- Close other tabs
- Check network latency
- Check API performance

**WebSocket not connecting:**
- Verify API supports WebSocket
- Check firewall rules
- Check proxy settings
- Verify ws:// protocol allowed

## Development

### Contributing UI Changes

1. Make changes in `web/`
2. Test in browser (F12 console)
3. Update `LAYOUT_CHANGES.md`
4. Submit PR

### Browser DevTools

```javascript
// Debug from console
window.apiUrl           // Current API URL
window.refreshInterval  // Update frequency
window.metrics          // Latest metrics

// Manual refresh
location.reload()

// Check API connection
fetch('http://localhost:8000/health')
```
