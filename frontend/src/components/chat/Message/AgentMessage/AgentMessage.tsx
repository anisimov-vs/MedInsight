import ReactMarkdown from 'react-markdown';
import './AgentMessage.css';
import { useUIStore } from '../../../../stores';

interface AgentMessageProps {
  id: string;
  text: string;
  timestamp: Date;
  chart?: { title: string; mode: 'line' | 'bar'; points: { month: string; cases: number }[] };
  plotlyData?: any;
}

const AgentMessage = ({ id, text, chart, plotlyData }: AgentMessageProps) => {
  const { openChart } = useUIStore();
  const hasChart = chart || plotlyData;

  return (
    <div className="agent-message">
      <div className="agent-message-content">
        <div className="agent-message-text">
          <ReactMarkdown>{text}</ReactMarkdown>
        </div>
        {hasChart && (
          <div className="agent-message-actions">
            <button className="agent-message-open-chart" onClick={() => openChart(id)}>Открыть график</button>
          </div>
        )}
      </div>
    </div>
  );
};

export default AgentMessage;
