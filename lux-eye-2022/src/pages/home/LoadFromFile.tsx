import { Code, Group, Text } from '@mantine/core';
import { Dropzone } from '@mantine/dropzone';
import { IconUpload } from '@tabler/icons';
import { useCallback } from 'react';
import { ErrorCode, FileRejection } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';
import { useStore } from '../../store';
import { notifyError } from '../../utils/notifications';
import { HomeCard } from './HomeCard';

function DropzoneContent(): JSX.Element {
  return (
    <Group position="center" spacing="xl" style={{ minHeight: 80, pointerEvents: 'none' }}>
      <IconUpload size={40}></IconUpload>
      <Text size="xl" inline={true}>
        Drag file here or click to select file
      </Text>
    </Group>
  );
}

export function LoadFromFile(): JSX.Element {
  const loadFromFile = useStore(state => state.loadFromFile);
  const loading = useStore(state => state.loading);

  const navigate = useNavigate();

  const onDrop = useCallback(
    (files: File[]) => {
      loadFromFile(files[0])
        .then(() => {
          navigate('/visualizer');
        })
        .catch((err: Error) => {
          console.error(err);
          notifyError(`Cannot load episode from ${files[0].name}`, err.message);
        });
    },
    [navigate],
  );

  const onReject = useCallback((rejections: FileRejection[]) => {
    for (const rejection of rejections) {
      notifyError(
        `Could not load episode from ${rejection.file.name}`,
        {
          [ErrorCode.FileInvalidType]: 'Invalid type, only HTML and JSON files are supported.',
          [ErrorCode.FileTooLarge]: 'File too large.',
          [ErrorCode.FileTooSmall]: 'File too small.',
          [ErrorCode.TooManyFiles]: 'Too many files.',
        }[rejection.errors[0].code]!,
      );
    }
  }, []);

  return (
    <HomeCard title="Load from file">
      {/* prettier-ignore */}
      <Text mb="xs">
        Supports JSON episodes generated using the <Code>luxai2022</Code> CLI or the <Code>kaggle-environments</Code> CLI.
      </Text>
      <Dropzone onDrop={onDrop} onReject={onReject} multiple={false} accept={['application/json']} loading={loading}>
        <Dropzone.Idle>
          <DropzoneContent />
        </Dropzone.Idle>
        <Dropzone.Accept>
          <DropzoneContent />
        </Dropzone.Accept>
      </Dropzone>
    </HomeCard>
  );
}
