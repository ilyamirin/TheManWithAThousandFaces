import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import TextField from '@material-ui/core/TextField';
import Typography from '@material-ui/core/Typography';
import Box from '@material-ui/core/Box';
import Autocomplete from '@material-ui/lab/Autocomplete';
import Paper from '@material-ui/core/Paper';
import InputAdornment from '@material-ui/core/InputAdornment';
import Button from '@material-ui/core/Button';

import object_cats from './categorical/objects.json';
import project_cats from './categorical/projects.json';
import nomenclature_cats from './categorical/nomenclatures.json';

const useStyles = makeStyles(theme => ({
  root: {
    display: 'flex',
    justifyContent: 'center',
    paddingTop: theme.spacing(4),
    height: '100%'
  },
  title: {
    paddingLeft: theme.spacing(1),
    marginBottom: theme.spacing(2)
  },
  alternativesDisplay: {
    paddingRight: theme.spacing(1)
  },
  inputDiv: {
    padding: theme.spacing(4, 3),
  },
  predictionPaper: {
    padding: theme.spacing(4, 3),
    height: '90%',
    backgroundColor: '#fafafa'
  },
  button: {
    marginTop: theme.spacing(2),
    marginRight: theme.spacing(1)
  }
}));


export default function App() {
  const classes = useStyles();

  const [inputVals, setInputVals] = React.useState({
    object: '',
    project: '',
    financing: '',
    nomenclature: '',
    description: ''
  });

  const [predVals, setPredVals] = React.useState({
    budget: null,
    turnover: null
  });


  const reqPredict = () => {
    fetch(`${process.env.REACT_APP_API_URL}/budget/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(inputVals)
    })
      .then(rawRes => rawRes.json())
      .then(res => setPredVals(prev => ({ ...prev, budget: res })))
    
    fetch(`${process.env.REACT_APP_API_URL}/turnover/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(inputVals)
    })
      .then(rawRes => rawRes.json())
      .then(res => setPredVals(prev => ({ ...prev, turnover: res })))
  }

  const clearInputs = () => {
    setInputVals({
      object: '',
      project: '',
      financing: '',
      nomenclature: '',
      description: ''
    })
    setPredVals({
      budget: null,
      turnover: null
    })
  }

  const getOuputHelpText = (predProbability) => {
    if (predProbability >= 95) {
      return '';
    } else if (predProbability > 70) {
      return "Невысокая уверенность модели. Возможна ошибка";
    } else {
      return "Большая вероятность ошибки. Рекомендуется ручная проверка"
    }
  }

  return (
    <Grid container spacing={2} className={classes.root}>
      <Grid item xs={6} sm={4}>
        <div className={classes.inputDiv}>
          <Typography variant="h5" className={classes.title} gutterBottom>Ввод</Typography>

          <Autocomplete
            options={object_cats}
            getOptionLabel={option => option}
            onChange={e => setInputVals({ ...inputVals, object: e.target.innerText })}
            value={inputVals.object}
            renderInput={params => (
              <TextField 
                {...params} 
                label="ЦФО"
                margin="normal" 
                variant="outlined" 
                fullWidth 
              />
            )}
          />

          <Autocomplete
            options={project_cats}
            getOptionLabel={option => option}
            onChange={e => setInputVals({ ...inputVals, project: e.target.innerText })}
            value={inputVals.project}
            renderInput={params => (
              <TextField 
                {...params} 
                label='Проект' 
                margin="normal" 
                variant="outlined" 
                fullWidth 
              />
            )}
          />

          <TextField
            label="ВЦС"
            margin="normal"
            variant="outlined" 
            fullWidth
            onChange={e => setInputVals({ ...inputVals, financing: e.target.value })}
            value={inputVals.financing}
          />

          <Autocomplete
            options={nomenclature_cats}
            getOptionLabel={option => option}
            onChange={e => setInputVals({ ...inputVals, nomenclature: e.target.innerText })}
            value={inputVals.nomenclature}
            renderInput={params => (
              <TextField 
                {...params} 
                label='Номенклатура' 
                margin="normal" 
                variant="outlined" 
                fullWidth 
              />
            )}
          />

          <TextField
            label="Описание"
            multiline
            fullWidth
            rows="4"
            rowsMax="6"
            margin="normal"
            variant="outlined"
            onChange={e => setInputVals({ ...inputVals, description: e.target.value })}
            value={inputVals.description}
          />

          <Button variant="outlined" color="secondary" className={classes.button} onClick={clearInputs}>
            Очистить
          </Button>

          <Button variant="contained" color="primary" className={classes.button} onClick={reqPredict}>
            Отправить
          </Button>
        </div>
      </Grid>
      <Grid item xs={6} sm={4}>
        <Paper className={classes.predictionPaper} style={{display: predVals.budget || predVals.turnover ? 'block' : 'none'}} elevation={3}>
          <Typography variant="h5" className={classes.title} gutterBottom>Результат</Typography>

          {predVals.turnover && <>
            <TextField
              error={predVals.turnover.main.probability < 70}
              label="Статья оборотов"
              helperText={getOuputHelpText(predVals.turnover.main.probability)}
              multiline
              fullWidth
              rows="1"
              rowsMax="6"
              margin="normal"
              InputProps={predVals.turnover.main.probability < 95 && {
                readOnly: true,
                endAdornment: <InputAdornment position="end">{predVals.turnover.main.probability}%</InputAdornment>,
              }}
              value={predVals.turnover.main.value}
            />
            {predVals.turnover.main.probability < 95 &&
              <Typography variant="body2" gutterBottom className={classes.alternativesDisplay}>
                <Box color="text.secondary">
                  <ul>
                    {predVals.turnover.alternatives.map(alt =>
                      <li>{alt.value} - {alt.probability}%</li>
                    )}
                  </ul>
                </Box>
              </Typography>
            }
          </>}

          {predVals.budget && <>
            <TextField
              error={predVals.budget.main.probability < 70}
              label="Смета"
              helperText={getOuputHelpText(predVals.budget.main.probability)}
              multiline
              fullWidth
              rows="1"
              rowsMax="6"
              margin="normal"
              InputProps={predVals.budget.main.probability < 95 && {
                readOnly: true,
                endAdornment: <InputAdornment position="end">{predVals.budget.main.probability}%</InputAdornment>,
              }}
              value={predVals.budget.main.value}
            />
            {predVals.budget.main.probability < 95 &&
              <Typography variant="body2" gutterBottom className={classes.alternativesDisplay}>
                <Box color="text.secondary">
                  <ul>
                    {predVals.budget.alternatives.map(alt =>
                      <li>{alt.value} - {alt.probability}%</li>
                    )}
                  </ul>
                </Box>
              </Typography>
            }
          </>}
        </Paper>
      </Grid>
    </Grid>
  );
}
